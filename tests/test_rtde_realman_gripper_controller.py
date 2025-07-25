import sys
import os

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Robotic_Arm.rm_robot_interface import *
from avp_stream import VisionProStreamer
import numpy as np
from scipy.spatial.transform import Rotation
import time

from diffusion_policy.real_world.rtde_realman_gripper_controller import RTDERealmanGripperController
from multiprocessing.managers import SharedMemoryManager

# Define a simple Low-Pass Filter class
class LowPassFilter:
    def __init__(self, alpha, initial_value):
        self.alpha = alpha
        self.filtered_value = np.array(initial_value, dtype=np.float64)

    def filter(self, new_value):
        self.filtered_value = self.alpha * np.array(new_value, dtype=np.float64) + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
    


def main():
    # 初始化参数
    robot_ip = "10.20.46.135"
    visionpro_ip = "10.12.169.155"
    vp_frequency = 20.0  # 目标频率 30Hz
    robot_frequency = 15.0
    joints_init = [1.341,20.622,-0.997,72.163,1.259,65.287,-0.68]  # 初始关节角度
    init_target_pose = [0.3536367416381836, 0.005998861975967884, 0.3363194465637207, 0.19016963243484497, -0.007337283343076706, 0.9816972017288208, 0.007230011280626059]

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

    # 使用共享内存管理器
    with SharedMemoryManager() as shm_manager:
        robot = RTDERealmanGripperController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=robot_frequency,
            joints_init=joints_init,
            init_target_pose=init_target_pose,
        )
        robot.start()
        robot.start_wait()
        print("RTDE VisionPro Controller started.")

        vp = VisionProStreamer(ip=visionpro_ip, record=True, frequency=vp_frequency)
        # Get API version
        print("\nAPI Version: ", rm_api_version(), "\n")
        time.sleep(1)   # 暂停1秒，等待VisionPro数据稳定

        init_robot_state = robot.get_state()
        print("Robot state:", init_robot_state)

        robot_initial_pose = init_robot_state['ActualTCPPose']
        print("robot_initial_pose: ", robot_initial_pose)
        robot_initial_pos = robot_initial_pose[:3] # 机械臂初始位置
        # robot_initial_rot_matrix = Rotation.from_euler('xyz', robot_initial_pose[3:]).as_matrix() # 机械臂初始旋转矩阵
        robot_initial_rot_matrix = Rotation.from_quat(robot_initial_pose[3:]).as_matrix() # 机械臂初始旋转矩阵

        # filter for position
        filter_alpha_pose = 0.7 # For position (x, y, z)
        lpf_pose = LowPassFilter(filter_alpha_pose, robot_initial_pos)

        # get hand initial pose
        base_hand_pose = vp.latest["left_wrist"].squeeze(0)
        print("the shape of base_ee_pose", base_hand_pose.shape)
        initial_hand_xyz = base_hand_pose[:3, 3]
        initial_hand_rotation_matrix = base_hand_pose[:3, :3]  #  for the after rotation transfor

        # 设置控制频率 (Hz)
        control_period = 1.0 / vp_frequency  # 每次循环的目标周期
        while True:
            loop_start_time = time.perf_counter()  # 记录循环开始时间
            # update visionpro data
            hand_pose_origin = vp.latest["left_wrist"].squeeze(0)
            hand_xyz = (hand_pose_origin[:3, 3] - initial_hand_xyz)
            hand_rotation_matrix = hand_pose_origin[:3, :3]

            # robot.realman_gripper(
            #     pose = [-0.001,0,0.85,1.0,0.0,0.0,0.0],
            #     close_gripper = 1,
            # )

            R_rel_vp = np.dot(hand_rotation_matrix, np.linalg.inv(initial_hand_rotation_matrix))    # 计算手部在 VP 坐标系中的相对旋转
            R_rel_robot = np.dot(R_Vp2Robot, np.dot(R_rel_vp, R_Vp2Robot.T))    # 将相对旋转转换到机械臂坐标系
            target_rot_matrix = np.dot(robot_initial_rot_matrix, R_rel_robot)   # 将转换后的相对旋转应用到机械臂初始旋转
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

            
            close_gripper = vp.latest["left_pinch_distance"] < 0.03
            # close_gripper = 1

            # target_pose = [0.2536367416381836, 0.005998861975967884, 0.2363194465637207, 0.19016963243484497, -0.007337283343076706, 0.9816972017288208, 0.007230011280626059]
            
            robot.realman_gripper(
                pose = target_pose,
                close_gripper = close_gripper,
            )


            robot_state = robot.get_state()
            # print("Robot state:", robot_state)

            # 计算实际耗时并补偿延迟
            loop_duration = time.perf_counter() - loop_start_time
            sleep_time = max(0.0, control_period - loop_duration)
            time.sleep(sleep_time)  # 精确等待剩余时间
            actual_freq = 1.0 / (loop_duration + sleep_time)
            # print(f"In test_rtde_vp_controller the Actual frequency: {actual_freq:.2f}Hz")


        time.sleep(1)

        robot.stop()
        print("RTDE VisionPro Controller stopped.")


if __name__ == "__main__":
    main()
