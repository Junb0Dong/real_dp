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

python eval_realman_test.py -i data/outputs/2025.08.01/16.34.06_train_diffusion_unet_image_realman_pick/checkpoints/latest.ckpt -o data/eval_pick_place_0801

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


OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, default="10.20.46.135", help="realman arm's IP address e.g. default = 10.20.46.135")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")


def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency):
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
    # 定义 frame_latency，基于相机帧率（假设 RealSense 相机为 30 FPS）
    frame_latency = 1 / 15  # 单位为秒，基于 30 FPS 相机帧率

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    joints_angle_init = [0.113,-8.014,0.185,87.687,-0.004,64.609,-0.008]
    pose_init = [0.25497639179229736, 0.0013807571958750486, 0.4147399663925171, 0.30669981241226196, -0.002576481783762574, 0.951802670955658, 0.0005165305337868631]

    with SharedMemoryManager() as shm_manager:
        with RealmanGripperEnv(
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
            env.realsense.set_exposure(exposure=120, gain=0)
            env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            obs = env.get_obs()
            print("Available observation keys:", obs.keys())  # 打印所有可用的观测键
            # 策略预热：获取一次观测，转换为模型输入格式，执行一次推理，确保模型能正常运行（避免首次推理延迟）。
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 8
                del result

            print('Ready!')
            episode_running = False  # 跟踪 episode 是否正在运行
            while True:                
                # 获取当前观测以用于可视化
                obs = env.get_obs()
                vis_img = obs[f'camera_{vis_camera_idx}'][-1]

                if episode_running:
                    # Episode 运行中的可视化
                    episode_id = env.replay_buffer.n_episodes
                    text = 'Episode: {}, Time: {:.1f}'.format(
                        episode_id, time.monotonic() - t_start
                    )
                else:
                    # 等待用户输入时的提示
                    text = 'Press C to start, Q to quit'

                # 显示图像和文字
                cv2.putText(
                    vis_img,
                    text,
                    (10, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    thickness=1,
                    color=(255, 255, 255)
                )
                cv2.imshow('default', vis_img[..., ::-1])

                # 检测键盘输入
                key_stroke = cv2.pollKey()
                if key_stroke == ord('q'):
                    print("Terminating program.")
                    if episode_running:
                        env.end_episode()
                    break

                if not episode_running:
                    # 等待用户启动 episode
                    if key_stroke == ord('c'):
                        print("Starting new episode!")
                        policy.reset()
                        start_delay = 1.0
                        eval_t_start = time.time() + start_delay
                        t_start = time.monotonic() + start_delay
                        env.start_episode(eval_t_start)
                        precise_wait(eval_t_start - frame_latency, time_func=time.time)
                        print("Started!")
                        episode_running = True
                        iter_idx = 0
                        term_area_start_timestamp = float('inf')
                        perv_target_pose = None
                    continue  # 继续循环等待键盘输入

                # ========== policy control loop ==============
                try:
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
                        action = result['action'][0].detach().to('cpu').numpy()
                        print('Inference latency:', time.time() - s)

                    # convert policy action to env actions
                    print("Policy action shape:", action.shape)
                    pose_action = action[:, :7]
                    gripper_action = np.round(action[:, 7]).astype(np.int32)
                    if delta_action:
                        assert action.shape[-1] == 8
                        if perv_target_pose is None:
                            perv_target_pose = obs['robot_eef_pose'][-1].copy()
                            assert perv_target_pose.shape == (7,)
                        this_target_pose = perv_target_pose + pose_action[-1]
                        perv_target_pose = this_target_pose.copy()
                        this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        gripper_action = gripper_action[[-1]]
                    else:
                        this_target_poses = np.zeros((len(pose_action), 7), dtype=np.float64)
                        this_target_poses[:] = pose_action
                    action_timestamps = (np.arange(len(pose_action), dtype=np.float64) + action_offset) * dt + obs_timestamps[-1]
                    action_exec_latency = 0.01
                    curr_time = time.time()
                    is_new = action_timestamps > (curr_time + action_exec_latency)

                    if np.sum(is_new) == 0:
                        this_target_poses = this_target_poses[[-1]]
                        gripper_action = gripper_action[[-1]]
                        next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                        action_timestamp = eval_t_start + (next_step_idx) * dt
                        print('Over budget', action_timestamp - curr_time)
                        action_timestamps = np.array([action_timestamp])
                    else:
                        this_target_poses = this_target_poses[is_new]
                        gripper_action = gripper_action[is_new]
                        action_timestamps = action_timestamps[is_new]

                    for i in range(len(this_target_poses)):
                        quat = this_target_poses[i, 3:7]
                        quat_norm = np.linalg.norm(quat)
                        if quat_norm > 1e-6:
                            this_target_poses[i, 3:7] = quat / quat_norm

                    gripper_state = bool(gripper_action[-1]) if len(gripper_action) > 0 else False
                    env.exec_actions(
                        actions=this_target_poses,
                        close_gripper_state=gripper_state,
                        timestamps=action_timestamps
                    )
                    print(f"Submitted {len(this_target_poses)} steps of actions.")

                    # 检查键盘输入
                    if key_stroke == ord('s'):
                        env.end_episode()
                        print('Stopped.')
                        episode_running = False
                        continue

                    # auto termination
                    if time.monotonic() - t_start > max_duration:
                        print('Terminated by timeout!')
                        env.end_episode()
                        episode_running = False
                        continue
                    else:
                        print("Out of the termination area.")
                        term_area_start_timestamp = float('inf')

                    # wait for execution
                    precise_wait(t_cycle_end - frame_latency)
                    iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.end_episode()
                    episode_running = False

                if not episode_running:
                    print("Stopped.")# 获取当前观测以用于可视化
    # 关闭 OpenCV 窗口
    cv2.destroyAllWindows()
# %%
if __name__ == '__main__':
    main()
