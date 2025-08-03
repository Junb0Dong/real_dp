"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

from diffusion_policy.real_world.realman_gripper_env import RealmanGripperEnv
from diffusion_policy.real_world.visionpro_shared_memory import VisionPro
from scipy.spatial.transform import Rotation


OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, default="10.20.46.135", help="realman arm's IP address e.g. default = 10.20.46.135")
@click.option('--vp_ip', '-ri', required=True, default="10.14.106.60", help="visionpro's IP address e.g. default = 10.12.169.155")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")


# Define a simple Low-Pass Filter class
class LowPassFilter:
    def __init__(self, alpha, initial_value):
        self.alpha = alpha
        self.filtered_value = np.array(initial_value, dtype=np.float64)

    def filter(self, new_value):
        self.filtered_value = self.alpha * np.array(new_value, dtype=np.float64) + (1 - self.alpha) * self.filtered_value
        return self.filtered_value

def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency,
    vp_ip):
    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    elif 'robomimic' in cfg.name:
        # BCRNN model
        policy: BaseImagePolicy
        policy = workspace.model

        device = torch.device('cuda')
        policy.eval().to(device)

        # BCRNN always has action horizon of 1
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        delta_action = cfg.task.dataset.get('delta_action', False)

    elif 'ibc' in cfg.name:
        policy: BaseImagePolicy
        policy = workspace.model
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096

        device = torch.device('cuda')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = 1
        delta_action = cfg.task.dataset.get('delta_action', False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    joints_angle_init = [0.113,-8.014,0.185,87.687,-0.004,64.609,-0.008]
    pose_init = [0.25497639179229736, 0.0013807571958750486, 0.4147399663925171, 0.30669981241226196, -0.002576481783762574, 0.951802670955658, 0.0005165305337868631]

    with SharedMemoryManager() as shm_manager:
        with VisionPro(shm_manager=shm_manager, frequency=30, visionpro_ip=vp_ip) as vp, \
            RealmanGripperEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                # recording resolution
                obs_image_resolution=(320,240),    # realsense 分辨率
                frequency=frequency, # default 10 Hz
                init_joints=joints_angle_init,
                target_pose_init=pose_init,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager
            ) as env:
            cv2.setNumThreads(1)

            # Should be the same as demo
            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            obs = env.get_obs()
            # 策略预热：获取一次观测，转换为模型输入格式，执行一次推理，确保模型能正常运行（避免首次推理延迟）。
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 2
                del result

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                target_pose = state['TargetTCPPose']
                t_start = time.monotonic()
                iter_idx = 0

                # VisionPro 参数
                vp_pose_scale_x = 0.9
                vp_pose_scale_y = 0.9
                vp_pose_scale_z = 0.9
                filter_alpha_pose = 0.7  # 低通滤波器参数
                # VisionPro 和机械臂坐标系转换矩阵
                R_Vp2Robot = np.array([
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, -1, 0]
                ])

                obs = env.get_obs() # 获取当前观测
                print("obs: ", obs.keys())
                # print("robot_initial_pose:", obs['robot_eef_pose'])
                # robot_initial_pose = obs['robot_state']['ActualTCPPose']
                robot_initial_pose = pose_init
                print("robot_initial_pose: ", robot_initial_pose)
                robot_initial_pos = robot_initial_pose[:3] # 机械臂初始位置
                robot_initial_rot_matrix = Rotation.from_quat(robot_initial_pose[3:]).as_matrix()

                # filter for position
                filter_alpha_pose = 0.7 # For position (x, y, z)
                lpf_pose = LowPassFilter(filter_alpha_pose, robot_initial_pos)

                time.sleep(1.0)

                # get hand initial pose
                base_hand_pose = vp.get_left_wrist_state()  # 获取左手腕状态
                print("the shape of base_ee_pose", base_hand_pose.shape)
                initial_hand_xyz = base_hand_pose[:3, 3]
                initial_hand_rotation_matrix = base_hand_pose[:3, :3]
                
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = np.minimum(vis_img, match_img)

                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    key_stroke = cv2.pollKey()
                    if key_stroke == ord('q'):
                        # Exit program
                        env.end_episode()
                        exit(0)
                    elif key_stroke == ord('c'):
                        # Exit human control loop
                        # hand control over to the policy
                        break

                    precise_wait(t_sample)

                    #TODO: get teleop command

                    hand_pose_origin = vp.get_left_wrist_state()  # 获取左手腕状态
                    hand_xyz = (hand_pose_origin[:3, 3] - initial_hand_xyz)
                    hand_rotation_matrix = hand_pose_origin[:3, :3]

                    R_rel_vp = np.dot(hand_rotation_matrix, np.linalg.inv(initial_hand_rotation_matrix)) 
                    R_rel_robot = np.dot(R_Vp2Robot, np.dot(R_rel_vp, R_Vp2Robot.T))
                    
                    euler_angles = Rotation.from_matrix(R_rel_robot).as_euler('ZYX')
                    adjusted_euler_angles = [-euler_angles[0], euler_angles[1], euler_angles[2]]    # 调整 roll 轴方向（取反 roll 角）
                    R_rel_robot_adjusted = Rotation.from_euler('ZYX', adjusted_euler_angles).as_matrix()    # 将调整后的欧拉角转换回旋转矩阵
                    target_rot_matrix = np.dot(robot_initial_rot_matrix, R_rel_robot_adjusted)  # 计算目标旋转矩阵
                    ee_quat_target = Rotation.from_matrix(target_rot_matrix).as_quat()  # 转换为四元数

                    d_pos_raw = hand_xyz[:3]
                    # 坐标系与机器人的坐标系对齐
                    d_pos_scaled = np.array([
                        d_pos_raw[1] * vp_pose_scale_y, # X_arm = Y_vp
                        d_pos_raw[0] * vp_pose_scale_x * -1, # Y_arm = -X_vp
                        d_pos_raw[2] * vp_pose_scale_z
                    ])

                    # position increment
                    ee_pos_target = robot_initial_pos + d_pos_scaled

                    # targetpose filter
                    filter_ee_pos_target = lpf_pose.filter(ee_pos_target)
                    target_pose = np.hstack([filter_ee_pos_target, ee_quat_target]) # test for quat
                    # print("pose_array: ",target_pose)
                    
                    # close_gripper = vp.get_left_pinch_distance() < 0.03
                    close_gripper = vp.get_right_pinch_distance() < 0.03
                    # close_gripper = 1

                    # print("close_gripper: ", close_gripper)
                    
                    # execute teleop command
                    env.exec_actions(
                        actions=[target_pose], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        close_gripper_state=close_gripper,
                        stages=[stage])
                    
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        print('get_obs')
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
                            print('Inference latency:', time.time() - s)

                        # convert policy action to env actions
                        print("Policy action shape:", action.shape)  # 应输出类似 (n_steps, 7) 或其他需要的维度
                        # convert policy action to env actions
                        if delta_action:
                            # 增量动作：action是相对于上一时刻的增量（假设维度为7）
                            assert action.shape[-1] == 7, f"Delta action must be 7D, got {action.shape[-1]}"
                            if perv_target_pose is None:
                                # 初始化为当前观测到的位姿（7维）
                                perv_target_pose = obs['robot_eef_pose'][-1].copy()
                                # 确保初始位姿是7维
                                assert perv_target_pose.shape == (7,), f"Initial pose must be 7D, got {perv_target_pose.shape}"
                            
                            # 增量叠加：上一时刻位姿 + 策略输出的增量
                            this_target_pose = perv_target_pose + action[-1]  # action[-1]取最后一步的增量
                            perv_target_pose = this_target_pose.copy()
                            this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        else:
                            # 绝对动作：策略直接输出7维目标位姿（x,y,z,qx,qy,qz,qw）
                            # 检查策略输出维度是否匹配
                            assert action.shape[-1] == 7, f"Absolute action must be 7D, got {action.shape[-1]}"
                            # 初始化目标位姿数组（形状：[n_steps, 7]）
                            this_target_poses = np.zeros((len(action), 7), dtype=np.float64)
                            # 直接使用策略输出的7维动作作为目标位姿
                            this_target_poses[:] = action  # 若策略输出完整7维，则直接赋值
                            # （可选）若策略只输出部分维度（如xyz），可保留原姿态：
                            # this_target_poses[:] = target_pose  # 先用当前目标位姿初始化
                            # this_target_poses[:, :3] = action[:, :3]  # 仅更新xyz，保持quat不变

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # # clip actions（裁剪安全范围，根据实际机器人工作空间调整）
                        # # 裁剪xyz位置（示例范围，需替换为你的机器人安全范围）
                        # this_target_poses[:, 0] = np.clip(this_target_poses[:, 0], 0.2, 0.8)  # x范围
                        # this_target_poses[:, 1] = np.clip(this_target_poses[:, 1], -0.5, 0.5)  # y范围
                        # this_target_poses[:, 2] = np.clip(this_target_poses[:, 2], 0.1, 0.5)   # z范围

                        # 确保四元数是单位四元数（避免姿态无效）
                        for i in range(len(this_target_poses)):
                            quat = this_target_poses[i, 3:7]  # 提取四元数 [qx, qy, qz, qw]
                            quat_norm = np.linalg.norm(quat)
                            if quat_norm > 1e-6:  # 避免除以0
                                this_target_poses[i, 3:7] = quat / quat_norm  # 归一化

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])


                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            env.end_episode()
                            print('Stopped.')
                            break

                        # auto termination
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        # term_pose = np.array([ 3.40948500e-01,  2.17721816e-01,  4.59076878e-02,  2.22014183e+00, -2.22184883e+00, -4.07186655e-04])
                        # curr_pose = obs['robot_eef_pose'][-1]
                        # dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
                        # if dist < 0.03:
                        #     # in termination area
                        #     curr_timestamp = obs['timestamp'][-1]
                        #     if term_area_start_timestamp > curr_timestamp:
                        #         term_area_start_timestamp = curr_timestamp
                        #     else:
                        #         term_area_time = curr_timestamp - term_area_start_timestamp
                        #         if term_area_time > 0.5:
                        #             terminate = True
                        #             print('Terminated by the policy!')
                        else:
                            # out of the area
                            print("Out of the termination area.")
                            term_area_start_timestamp = float('inf')

                        if terminate:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
