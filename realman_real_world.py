"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation
from diffusion_policy.real_world.realman_gripper_env import RealmanGripperEnv
from diffusion_policy.real_world.visionpro_shared_memory import VisionPro
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)


# Define a simple Low-Pass Filter class
class LowPassFilter:
    def __init__(self, alpha, initial_value):
        self.alpha = alpha
        self.filtered_value = np.array(initial_value, dtype=np.float64)

    def filter(self, new_value):
        self.filtered_value = self.alpha * np.array(new_value, dtype=np.float64) + (1 - self.alpha) * self.filtered_value
        return self.filtered_value


@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, default="10.20.46.135", help="realman arm's IP address e.g. default = 10.20.46.135")
@click.option('--vp_ip', '-ri', required=True, default="10.14.106.60", help="visionpro's IP address e.g. default = 10.12.169.155")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
# @click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, vis_camera_idx, frequency, command_latency, vp_ip):
    dt = 1/frequency    # delta t = 1/frequency
    # 一代关节数据
    # joints_init = [1.341,20.622,-0.997,72.163,1.259,65.287,-0.68]  # 初始关节角度
    # init_target_pose = [0.3536367416381836, 0.005998861975967884, 0.3363194465637207, 0.19016963243484497, -0.007337283343076706, 0.9816972017288208, 0.007230011280626059]
    # 二代关节数据
    # joint_init = [-56.13,31.024,-0.708,58.763,0.892,80.452,-0.482],[-0.97963588, 0.54138366, -0.01235693, 1.02560784, 0.01560324, 1.40420468, -0.00844739]
    # init_target_pose = [0.20320788025856018, -0.3048255741596222, 0.318773090839386, 0.0769142135977745, 0.4700179398059845, 0.8785333633422852, -0.03669378161430359]
    # 三代关节数据
    # joint_init = [-54.858,-0.45,0.001,87.28,0.001,64.248,-30.001], [-9.57435241e-01, -7.83652836e-03, -1.74532933e-05, 1.52325358e+00, 6.98131734e-05, 1.12135660e+00, 5.23633696e-01]
    # init_target_pose = [0.1596326380968094, -0.22676502168178558, 0.3820781111717224, 0.18435519933700562, 0.2083999514579773, 0.9456217288970947, -0.1684700846672058]
    joints_angle_init = [0.113,-8.014,0.185,87.687,-0.004,64.609,-0.008]
    pose_init = [0.25497639179229736, 0.0013807571958750486, 0.4147399663925171, 0.30669981241226196, -0.002576481783762574, 0.951802670955658, 0.0005165305337868631]
    with SharedMemoryManager() as shm_manager:  # 共享内存，键盘监听，SpaceMouse输入
        with KeystrokeCounter() as key_counter, \
            VisionPro(shm_manager=shm_manager, frequency=30, visionpro_ip=vp_ip) as vp, \
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

            print("enter the 1st step of main funcution!")


            time.sleep(3.0)

            cv2.setNumThreads(1)
            

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)
            
            
            # prepare for vp teleoperation 
            # VisionPro 参数
            vp_pose_scale_x = 0.8
            vp_pose_scale_y = 0.8
            vp_pose_scale_z = 0.8
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
            robot_initial_rot_matrix = Rotation.from_quat(robot_initial_pose[3:]).as_matrix() # 机械臂初始旋转矩阵

            # filter for position
            filter_alpha_pose = 0.7 # For position (x, y, z)
            lpf_pose = LowPassFilter(filter_alpha_pose, robot_initial_pos)

            print("enter the 2nd step of main funcution!")
           

            time.sleep(1.0)

            # get hand initial pose
            base_hand_pose = vp.get_left_wrist_state()  # 获取左手腕状态
            print("the shape of base_ee_pose", base_hand_pose.shape)
            initial_hand_xyz = base_hand_pose[:3, 3]
            initial_hand_rotation_matrix = base_hand_pose[:3, :3]


            print('Ready!')
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose']
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt # 本轮循环结束时间
                t_sample = t_cycle_end - command_latency    # 采样输入的目标时间
                t_command_target = t_cycle_end + dt # 命令执行的目标时间

                # pump obs
                obs = env.get_obs() # 获取当前观测

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                stage = key_counter[Key.space]

                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy() #  获取当前相机视角的图像
                episode_id = env.replay_buffer.n_episodes #  获取当前回放缓冲区中的episode数量
                text = f'Episode: {episode_id}, Stage: {stage}' #  构造文本信息
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()   #  获取键盘按键

                precise_wait(t_sample)  # 精确等待到当前周期结束，应该是要把state-action和obs对齐
                
                # get visionpro data to teleoperate the robot
                # update visionpro data
                hand_pose_origin = vp.get_left_wrist_state()  # 获取左手腕状态
                hand_xyz = (hand_pose_origin[:3, 3] - initial_hand_xyz)
                hand_rotation_matrix = hand_pose_origin[:3, :3]

                R_rel_vp = np.dot(hand_rotation_matrix, np.linalg.inv(initial_hand_rotation_matrix))    # 计算手部在 VP 坐标系中的相对旋转
                R_rel_robot = np.dot(R_Vp2Robot, np.dot(R_rel_vp, R_Vp2Robot.T))    # 将相对旋转转换到机械臂坐标系
                # target_rot_matrix = np.dot(robot_initial_rot_matrix, R_rel_robot)   # 将转换后的相对旋转应用到机械臂初始旋转
                # ee_quat_target = Rotation.from_matrix(target_rot_matrix).as_quat()  # 转换为四元数

                
                # 将 R_rel_robot 转换为欧拉角（假设 'ZYX' 顺序：yaw-Z, pitch-Y, roll-X）
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

                
                close_gripper = vp.get_left_pinch_distance() < 0.03
                # close_gripper = 1

                # print("close_gripper: ", close_gripper)
                
                # execute teleop command
                env.exec_actions(
                    actions=[target_pose], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    close_gripper_state=close_gripper,
                    stages=[stage])
                precise_wait(t_cycle_end)   # 精确等待到下一周期开始
                iter_idx += 1   # 循环计数加1

# %%
if __name__ == '__main__':
    main()
