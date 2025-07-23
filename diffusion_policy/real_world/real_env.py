from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator, 
    TimestampActionAccumulator,
    align_timestamps
)
from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)

DEFAULT_OBS_KEY_MAP = {
    # robot
    'ActualTCPPose': 'robot_eef_pose',
    'ActualTCPSpeed': 'robot_eef_pose_vel',
    'ActualQ': 'robot_joint',
    'ActualQd': 'robot_joint_vel',
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}

class RealEnv:
    def __init__(self, 
            # required params
            output_dir, # 输出目录，在demo_real_robot.py中args来指定
            robot_ip,   # robot ip->UR5
            # env params
            frequency=10,   # 控制频率
            n_obs_steps=2,  # 观测步骤
            # obs
            obs_image_resolution=(640,480), # image分辨率
            max_obs_buffer_size=30, # 最大观测缓冲区大小
            camera_serial_numbers=None,
            obs_key_map=DEFAULT_OBS_KEY_MAP,    # obs观测字典
            obs_float32=False,
            # action
            max_pos_speed=0.25, # robot最大的position运动速度
            max_rot_speed=0.6,  # robot最大的rotation运动速度
            # robot
            tcp_offset=0.13,    # 机器人末端执行器的位置和姿态偏移
            init_joints=False,  # joint 初始化
            # video capture params
            video_capture_fps=30,   # video的fps
            video_capture_resolution=(1280,720),    # NOTE video的分辨率，为什么比上面大？video_capture_resolution作为输入分辨率，obs_image_resolution作为输出分辨率
            # saving params
            record_raw_video=True,
            thread_per_video=2, # 每个视频流分配的线程数（用于编码/处理）
            video_crf=21,   # 视频压缩率因子（CRF值，范围0-51，越低质量越高）
            # vis params
            enable_multi_cam_vis=True,  # 是否启用多摄像头画面拼接显示
            multi_cam_vis_resolution=(1280,720),    # 多摄像头拼接画面的分辨率（宽×高）
            # shared memory
            shm_manager=None    # 共享内存管理器，一个用到相机上，一个用到robot上
            ):
        assert frequency <= video_capture_fps   # frequency must be less than or equal to video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')  # 打开zarr文件

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start() # 创建一个共享内存管理器进程
        if camera_serial_numbers is None:   # 相机序列号
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution, 
            # obs output rgb
            bgr_to_rgb=True)    # 转换图像的分辨率和颜色格式（BGR->RGB）
        color_transform = color_tf
        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):
            data['color'] = color_transform(data['color'])
            return data
        
        rw, rh, col, row = optimal_row_cols( #  计算最优的行和列数
            n_cameras=len(camera_serial_numbers), #  相机数量
            in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1], #  观测图像的宽高比
            max_resolution=multi_cam_vis_resolution #  最大分辨率
        )
        vis_color_transform = get_image_transform( #  获取图像变换
            input_res=video_capture_resolution, #  输入分辨率
            output_res=(rw,rh), #  输出分辨率
            bgr_to_rgb=False #  是否将BGR转换为RGB
        )
        def vis_transform(data):    # 对相机捕获的图像数据进行特定的转换操作，以满足可视化的需求
            data['color'] = vis_color_transform(data['color'])
            return data

        # video recording
        recording_transfrom = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_transfrom = transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps, 
            codec='h264',
            input_pix_fmt=recording_pix_fmt, 
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)

        # 相机初始化
        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers, # 自动获取序列号
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=False, # 不需要深度信息
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,5
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transfrom,
            video_recorder=video_recorder,
            verbose=False
            )
        
        # 相机可视化，独立进程
        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )


        cube_diag = np.linalg.norm([1,1,1])
        j_init = np.array([0,-90,-90,-90,90,0]) / 180 * np.pi
        if not init_joints:
            j_init = None

        # 创建RTDE对象用于控制robot，在独立进程中运行，并通过共享内存 (SharedMemoryManager) 与其他进程通信
        robot = RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=125, # UR5 CB3 RTDE
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed*cube_diag, # 防止机器人超出工作空间
            max_rot_speed=max_rot_speed*cube_diag,
            launch_timeout=3,
            tcp_offset_pose=[0,0,tcp_offset,0,0,0], # 工具中心点位姿 前三个元素是xyz坐标，后三个元素是欧拉角
            payload_mass=None,  # 负载
            payload_cog=None,
            joints_init=j_init, # 关节初始位置
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size
            )
        self.realsense = realsense
        self.robot = robot
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready
    
    # 启动相机、机器人和可视化组件，并根据 wait 参数决定是否等待它们启动完成。
    def start(self, wait=True):
        self.realsense.start(wait=False)
        self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    # 停止相机、机器人和可视化组件，并根据 wait 参数决定是否等待它们停止完成
    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    # 启动相机、机器人和可视化组件，并等待它们启动完成
    def start_wait(self):
        self.realsense.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency)) # math.ceil：向上取整
        self.last_realsense_data = self.realsense.get(k=k, out=self.last_realsense_data) # 获取k个数据

        # 125 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()    # 从共享内存中获取robot状态
        # both have more than n_obs_steps data

        # align camera obs timestamps 从robot的最后一个时间戳开始，向前推n_obs_steps个时间戳, align：对齐
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()])    # 获取所有摄像头最新时间戳中的最大值
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)    # 生成一组倒退对齐的时间戳序列
        # 结果示例，[::-1]：反转数组 → [n_obs_steps-1, ..., 2, 1, 0]
        # [10.0 - 2*0.1,   → 9.8
        # 10.0 - 1*0.1,   → 9.9
        # 10.0 - 0*0.1]   → 10.0

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0] # 找到当前摄像头所有早于对齐时间戳t的帧索引
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            camera_obs[f'camera_{camera_idx}'] = value['color'][this_idxs]

        # align robot obs
        robot_timestamps = last_robot_data['robot_receive_timestamp']
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v
        
        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps
            )

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data
    
    # 执行新的action并保存到共享内存中
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray, 
            stages: Optional[np.ndarray]=None):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # schedule waypoints
        for i in range(len(new_actions)): # 执行新动作
            self.robot.schedule_waypoint(
                pose=new_actions[i],
                target_time=new_timestamps[i]
            )
        
        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )
    
    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            if n_steps > 0:
                episode = dict()
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')

