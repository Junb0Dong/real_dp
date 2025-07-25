import sys
import os
import enum

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Robotic_Arm.rm_robot_interface import *
import numpy as np
import multiprocessing
import time
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_queue import (SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


# TODO：准备先把采集数据的框架搭出来，后面数据质量不高的话再应用PoseTrajectoryInterpolator

class Command(enum.Enum):
    STOP = 0
    REALMAN_GRIPPER = 3   # 调度夹爪运动

# Define a simple Low-Pass Filter class
class LowPassFilter:
    def __init__(self, alpha, initial_value):
        self.alpha = alpha
        self.filtered_value = np.array(initial_value, dtype=np.float64)

    def filter(self, new_value):
        self.filtered_value = self.alpha * np.array(new_value, dtype=np.float64) + (1 - self.alpha) * self.filtered_value
        return self.filtered_value


# gripper control subprocess
def gripper_control_proc(gripper_state, robot_controller):
    """子进程：根据gripper_state实时控制夹爪"""
    # last_state = None
    while True:
        if gripper_state.value == 1:
            robot_controller.rm_set_gripper_pick(500, 200, True, 10)
        else:
            robot_controller.rm_set_gripper_release(500, True, 10)
        # last_state = gripper_state.value

class RTDERealmanGripperController(mp.Process):
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        robot_ip = "10.20.46.135", 
        frequency=30, 
        joints_init=None,
        receive_keys=None,
        get_max_k=128,
        init_target_pose=None,
        ):
        
        # verify
        assert 0 < frequency <= 100
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="RTDEPositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.joints_init = joints_init
        self.init_target_pose = init_target_pose

        # build input queue
        example = {
            'cmd': Command.REALMAN_GRIPPER.value,
            'target_pose': np.zeros((7,), dtype=np.float64),    # NOTE：这里需要改一下？改成7（xyz+quat）
            'close_gripper': bool(False)
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
                'ActualQ',  # 各关节的实际角度

                'TargetTCPPose',    # 只能用visionpro的来替代
                'gripper_state',    # 夹爪状态
            ]

        example = {
            "ActualQ":  np.zeros(7, dtype=np.float64),
            "ActualTCPPose": np.zeros(7, dtype=np.float64),
            "TargetTCPPose": np.zeros(7, dtype=np.float64),
            "gripper_state": bool(False),
            "robot_receive_timestamp": time.time()    # 添加时间戳
        }

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.current_target_pose = np.array(self.init_target_pose, dtype=np.float64) # Placeholder
        print("first current target pose is also init target pose: ", self.current_target_pose)
        self.close_gripper = False

        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

        print("this is __init__ function of RTDERealmanGripperController")

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        """Stop the process and optionally wait for it to terminate."""
        self.stop_event.set()
        if wait:
            self.join()

    def start_wait(self):
        self.ready_event.wait(3)
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
    def realman_gripper(self, pose, close_gripper):
        pose = np.array(pose)
        assert pose.shape == (7,)   # 使用xyz+quat表示位姿

        message = {
            'cmd': Command.REALMAN_GRIPPER.value,
            'target_pose': pose,
            'close_gripper': close_gripper
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
        # 在主进程中创建唯一的 robot 和 algo_handle 实例
        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.arm_model = rm_robot_arm_model_e.RM_MODEL_RM_75_E
        self.force_type = rm_force_type_e.RM_MODEL_RM_B_E
        self.algo_handle = Algo(self.arm_model, self.force_type)

        try:
            handle = self.robot.rm_create_robot_arm(self.robot_ip, 8080)
            print("机械臂ID：", handle.id)

            # Define state getters AFTER robot is initialized in THIS process
            self.state_getters = {
                "ActualQ": lambda: np.array(self.robot.rm_get_joint_degree()[1]),
                "ActualTCPPose": lambda: np.array(self.algo_handle.rm_algo_forward_kinematics(self.robot.rm_get_joint_degree()[1], 0)),
                "TargetTCPPose": lambda: np.array(self.current_target_pose),
                "gripper_state": lambda: bool(self.close_gripper),
            }


            # init pose
            if self.joints_init is not None:
                print("this rm_move_j function")
                print(self.robot.rm_movej(self.joints_init, v=20, r=0, connect=0, block=1)) # 移动到初始位姿
                # time.sleep(3)


            # gripper control
            gripper_state = multiprocessing.Value('i', 0)  # 0: open, 1: close
            # gripper_proc = multiprocessing.Process(target=gripper_control_proc, args=(gripper_state, self.robot))
            # gripper_proc.daemon = True 
            # gripper_proc.start()

            self._last_gripper_commanded_state = open # None, "open", "close"
            print("this is run function of RTDERealmanGripperController")


            time.sleep(1)
            print("Enter the run LOOOOOOOOP of run")

            iter_idx = 0
            control_period = 1.0 / self.frequency  # 每次循环的目标周期
            keep_running = True
            print(f"stop_event initial state: {self.stop_event.is_set()}")
            while keep_running and not self.stop_event.is_set():
                print(f"Loop iteration {iter_idx}: stop_event={self.stop_event.is_set()}, keep_running={keep_running}")
                loop_start_time = time.perf_counter()  # 记录循环开始时间

                self.robot.rm_movep_canfd(self.current_target_pose, False, 1, 80)   # robot move to target pose
                # self.robot.rm_movej_p(self.current_target_pose, v=20, r=0, connect=0, block=1)  # robot move to target pose
                # gripper_state.value = 1 if self.close_gripper else 0    # update gripper state

                if self.close_gripper:
                    if self._last_gripper_commanded_state != "close":
                        self.robot.rm_set_gripper_pick(500, 200, True, 10) # 仍然是阻塞的
                        self._last_gripper_commanded_state = "close"
                else:
                    if self._last_gripper_commanded_state != "open":
                        self.robot.rm_set_gripper_release(500, True, 10) # 仍然是阻塞的
                        self._last_gripper_commanded_state = "open"

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()   # NOTE: get command from shared memory
                    print("commands['cmd']:", commands['cmd'], type(commands['cmd']), np.shape(commands['cmd']))
                    n_cmd = len(commands['cmd'])
                except Empty:
                    print("No commands received.")
                    n_cmd = 0 

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    print(f"Processing command {i+1}/{n_cmd}: cmd={cmd}")
                    if cmd == Command.STOP.value:
                        print("Received STOP command, stopping the loop.")
                        keep_running = False
                        # stop immediately, ignore later commands
                        break

                    elif cmd == Command.REALMAN_GRIPPER.value:
                        print(f"Received REALMAN_GRIPPER command: target_pose={command['target_pose']}, close_gripper={command['close_gripper']}")
                        self.current_target_pose = command['target_pose']
                        self.close_gripper = command['close_gripper']

                # update robot state
                state = dict()
                for key in self.receive_keys:
                    try:
                        state[key] = self.state_getters[key]() # Get robot state using the instance in this process
                    except Exception as e:
                        print(f"Error getting state for key {key}: {e}")
                        # Provide a placeholder or default value if state reading fails
                        if key == "ActualQ": state[key] = np.zeros(7, dtype=np.float64)
                        elif key == "ActualTCPPose": state[key] = np.zeros(7, dtype=np.float64)
                        elif key == "TargetTCPPose": state[key] = np.zeros(7, dtype=np.float64)
                        elif key == "gripper_state": state[key] = False

                state['robot_receive_timestamp'] = time.time()
                self.ring_buffer.put(state) # 存储到共享内存中
                
                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # 计算实际耗时并补偿延迟
                loop_duration = time.perf_counter() - loop_start_time
                sleep_time = max(0.0, control_period - loop_duration)
                time.sleep(sleep_time)  # 精确等待剩余时间
                actual_freq = 1.0 / (loop_duration + sleep_time)
                print(f"Actual frequency: {actual_freq:.2f}Hz")

        except Exception as e:
            print(f"An unexpected error occurred in RTDERealmanGripperController run loop: {e}")
        finally:
            # Mandatory cleanup - only if robot was successfully created
            if self.robot:
                try:
                    # IMPORTANT: Send a stop command to the robot here
                    print("RTDERealmanGripperController: Sending stop command to robot.")
                    # Also consider if there's a command to clear alarms or reset state
                    # self.robot.rm_clear_alarms() # If applicable and needed
                    self.robot.rm_delete_robot_arm()
                    self.robot.rm_destroy()
                    print("RTDERealmanGripperController: Robot arm instance deleted.")
                except Exception as e:
                    print(f"Error during robot cleanup: {e}")
            # Ensure ready event is set even if an error occurred, to unblock parent
            self.ready_event.set() # This might already be set from iter_idx == 0, but safe to repeat.
            print("RTDERealmanGripperController process terminated.")
