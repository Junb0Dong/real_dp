import sys
import os

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Robotic_Arm.rm_robot_interface import *
from avp_stream import VisionProStreamer
import numpy as np
from scipy.spatial.transform import Rotation
import multiprocessing
import time



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

def main():
    # time.sleep(6)
    # Create a robot arm controller instance and connect to the robot arm
    # robot_controller = RobotArmController("10.20.46.135", 8080, 3)
    robot_ip = "10.20.46.135"
    robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = robot.rm_create_robot_arm(robot_ip, 8080)
    print("机械臂ID：", handle.id)

    # init visionpro
    visionpro_ip = "10.12.169.155" 
    vp_pose_scale_x = 0.8
    vp_pose_scale_y = 0.8
    vp_pose_scale_z = 0.8
    vp = VisionProStreamer(ip=visionpro_ip, record=True, frequency=40)
    time.sleep(1)   # 暂停0.5秒，等待VisionPro数据稳定

    # Rotatation matrix for vp2robot
    R_Vp2Robot = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, -1, 0] 
        ])

    # Get API version
    print("\nAPI Version: ", rm_api_version(), "\n")

    # Perform movej motion
    robot.rm_movej([1.341,20.622,-0.997,72.163,1.259,65.287,-0.68], v=20, r=0, connect=0, block=1)

    # robot arm joint2pose solver
    arm_model = rm_robot_arm_model_e.RM_MODEL_RM_75_E
    force_type = rm_force_type_e.RM_MODEL_RM_B_E
    algo_handle = Algo(arm_model, force_type)

    joint_angle = robot.rm_get_joint_degree()[1]
    print(joint_angle)
    robot_initial_pose = algo_handle.rm_algo_forward_kinematics(joint_angle, 0)
    print("robot_initial_pose: ", robot_initial_pose)
    robot_initial_pos = robot_initial_pose[:3] # 机械臂初始位置
    # robot_initial_rot_matrix = Rotation.from_euler('xyz', robot_initial_pose[3:]).as_matrix() # 机械臂初始旋转矩阵
    robot_initial_rot_matrix = Rotation.from_quat(robot_initial_pose[3:]).as_matrix() # 机械臂初始旋转矩阵

    # get hand initial pose
    base_hand_pose = vp.latest["left_wrist"].squeeze(0)
    print("the shape of base_ee_pose", base_hand_pose.shape)
    initial_hand_xyz = base_hand_pose[:3, 3]
    initial_hand_rotation_matrix = base_hand_pose[:3, :3]  #  for the after rotation transfor

    # gripper control
    gripper_state = multiprocessing.Value('i', 0)  # 0: open, 1: close
    gripper_proc = multiprocessing.Process(target=gripper_control_proc, args=(gripper_state, robot))
    gripper_proc.daemon = True 
    gripper_proc.start()

    # filter for position
    filter_alpha_pose = 0.7 # For position (x, y, z)
    lpf_pose = LowPassFilter(filter_alpha_pose, robot_initial_pos)

    # 设置控制频率 (Hz)
    target_frequency = 30.0  # 例如30Hz
    control_period = 1.0 / target_frequency  # 每次循环的目标周期

    while True:
        loop_start_time = time.perf_counter()  # 记录循环开始时间

        # update visionpro data
        hand_pose_origin = vp.latest["left_wrist"].squeeze(0)
        hand_xyz = (hand_pose_origin[:3, 3] - initial_hand_xyz)
        hand_rotation_matrix = hand_pose_origin[:3, :3]

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

        # execute motion
        robot.rm_movep_canfd(target_pose, False, 1, 80)

        close_gripper = vp.latest["left_pinch_distance"] < 0.03
        gripper_state.value = 1 if close_gripper else 0

        # 计算实际耗时并补偿延迟
        loop_duration = time.perf_counter() - loop_start_time
        sleep_time = max(0.0, control_period - loop_duration)
        time.sleep(sleep_time)  # 精确等待剩余时间
        actual_freq = 1.0 / (loop_duration + sleep_time)
        print(f"Actual frequency: {actual_freq:.2f}Hz")

if __name__ == "__main__":
    main()