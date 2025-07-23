import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
from rtde_control import RTDEControlInterface   # 这块是colse source的，只提供了python接口
from rtde_receive import RTDEReceiveInterface
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1  # 线性伺服运动
    SCHEDULE_WAYPOINT = 2   # 调度路径点


class RTDEInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip, 
            frequency=125, 
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,
            ):
        """
         : CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose    工具中心点位姿 前三个元素是xyz坐标，后三个元素是欧拉角
        payload_mass: float 负载质量
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="RTDEPositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64), # 改成ti5,只需要`target_pose`? 这看是否在`demo_real_robot.py`中使用了duration和target_time
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer：环形缓冲区
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',    # 末端执行器的中心位姿
                'ActualTCPSpeed',   # 末端执行器的速度
                'ActualQ',  # 各关节的实际角度
                'ActualQd', # 各关节的实际速度

                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd'
            ]
        rtde_r = RTDEReceiveInterface(hostname=robot_ip)
        example = dict()
        for key in receive_keys:
            example[key] = np.array(getattr(rtde_r, 'get'+key)())   # 配合ur_rtde的API
        example['robot_receive_timestamp'] = time.time()    # 添加时间戳
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
    
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)    # 得有put才会有get
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start rtde
        robot_ip = self.robot_ip
        rtde_c = RTDEControlInterface(hostname=robot_ip)    # client send command to robot
        rtde_r = RTDEReceiveInterface(hostname=robot_ip)    # client receive state from robot

        try:
            if self.verbose:     #  如果verbose为True，则打印连接到机器人的信息
                print(f"[RTDEPositionalController] Connect to robot: {robot_ip}")

            # set parameters
            if self.tcp_offset_pose is not None:
                rtde_c.setTcp(self.tcp_offset_pose)     # 工具中心点位姿
            if self.payload_mass is not None:
                if self.payload_cog is not None:
                    assert rtde_c.setPayload(self.payload_mass, self.payload_cog)
                else:
                    assert rtde_c.setPayload(self.payload_mass)
            
            # init pose
            if self.joints_init is not None:
                assert rtde_c.moveJ(self.joints_init, self.joints_init_speed, 1.4)  # 移动到初始位姿

            # main loop
            dt = 1. / self.frequency
            curr_pose = rtde_r.getActualTCPPose()
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            # 位姿轨迹插值器（PoseTrajectoryInterpolator）
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )
            
            iter_idx = 0
            keep_running = True
            while keep_running:
                # start control iteration
                t_start = rtde_c.initPeriod()   # start of a control period / cycle

                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                pose_command = pose_interp(t_now)   # 使用插值法获取当前时刻的目标位姿
                vel = 0.5
                acc = 0.5
                # 将计算出的目标位姿传递给机器人
                assert rtde_c.servoL(pose_command, 
                    vel, acc, # dummy, not used by ur5
                    dt, 
                    self.lookahead_time, 
                    self.gain)
                
                # update robot state
                state = dict()
                for key in self.receive_keys:
                    state[key] = np.array(getattr(rtde_r, 'get'+key)()) # 获取robot state
                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state) # 存储到共享内存中

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']    # 目标位姿
                        duration = float(command['duration'])   # 期望完成时间
                        curr_time = t_now + dt  # 当前控制周期的“有效时间”（考虑执行延迟）
                        t_insert = curr_time + duration # 期望到达新航点的时间=curr_time + duration
                        # 从当前轨迹在 curr_time 的位姿开始，到 t_insert 时到达 target_pose，中间通过插值生成连续位姿
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,   # 新目标位姿
                            time=t_insert,      # 期望到达时间
                            curr_time=curr_time,    # 当前时间（轨迹起点时间）
                            max_pos_speed=self.max_pos_speed,   # 最大移动速度限制
                            max_rot_speed=self.max_rot_speed    # 最大旋转速度限制
                        )
                        last_waypoint_time = t_insert   # 最后一个waypoint的时间
                        if self.verbose:
                            print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # regulate frequency
                rtde_c.waitPeriod(t_start)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # manditory cleanup
            # decelerate
            rtde_c.servoStop()

            # terminate
            rtde_c.stopScript()
            rtde_c.disconnect()
            rtde_r.disconnect()
            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {robot_ip}")
