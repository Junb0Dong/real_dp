"""
visionpro_scoket.py的单元测试
将visionpro的数据发送到共享内存后，从共享内存中读取数据并发送给机械臂
模仿UR机械臂的rtde_c
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.visionpro_shared_memory import VisionPro
# socket
import socket
import json
from diffusion_policy.socket.robot_receiver import RobotReceiver
from diffusion_policy.socket.robot_controller import RobotController

import math
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':
    # 机器人 IP 和端口
    HOST = "10.20.55.106"  # 替换为实际机器人 IP
    # HOST = "127.0.0.1"  # 本地测试
    PORT = 9000
    sender = RobotController(HOST, PORT)# 创建发送类实例
    receiver = RobotReceiver(sender.get_socket())# 创建接收类实例，传递共享 socket

    vp_ip="10.12.174.92"    # visionpro 的 ip

    # bias_xyz = np.array([2.79187e-5, -0.49621, 0.362338])  # xyz偏置
    # q = [0.148742, -5.05746e-5, 2.28309e-5, 0.988876]   # 四元数偏置，标量最后顺序xyzw

    # #Avoid singularities
    # bias_xyz = np.array([171.389, -364.496, 420.007])  # xyz偏置
    # bisa_rpy = [-0.0878895, 1.4993, 0.631517]   # 四元数偏置，标量最后顺序xyzw
    # # #push T 
    # bias_xyz = np.array([315.765, -388.611, 168.189])  # xyz偏置
    # bisa_rpy = [-2.82106, -3.1295, 0.00854301]   # 四元数偏置，标量最后顺序xyzw
    # new sdk test
    bias_xyz = np.array([78.0628, -366.38, 240.631])  # xyz偏置
    bisa_rpy = [0.209908, -0.000148732, 1.99512]   # 四元数偏置，标量最后顺序xyzw

    vp_pose_scale = 1000  # 视觉数据缩放比例

    print("Running VisionPro test...")
    with SharedMemoryManager() as shm_manager:
        with VisionPro(shm_manager=shm_manager, frequency=10, dtype=np.float64, visionpro_ip=vp_ip) as vp:
            time.sleep(1.0)
            print("VisionPro started.")

            frequency = 0.5  # 10 Hz
            interval = 1.0 / frequency  # Time interval between iterations
            
            # 设置base pose
            basepose = vp.get_right_wrist_state()
            base_xyz = basepose[:3, 3]*vp_pose_scale
            base_rotation_matrix = basepose[:3, :3]


            # 将四元数转换为旋转矩阵
            # bias_rotation_matrix = R.from_quat(q).as_matrix()

            # 将rpy转换为旋转矩阵
            bias_rotation_matrix = R.from_euler('xyz', bisa_rpy, degrees=False).as_matrix()

            while True:
                # print("111111111111111111111111111111111")
                start_time = time.time()
                transform_matrix = vp.get_right_wrist_state()  # 获取左手腕状态
                # print("transform_matrix:", transform_matrix)
                xyz = transform_matrix[:3, 3]*vp_pose_scale  # 提取平移部分 (x, y, z)
                # 计算相对位置
                xyz = xyz - base_xyz  + bias_xyz # 相对位置
                # print("相对位置：", xyz)
                rotation_matrix = transform_matrix[:3, :3]  # 提取旋转矩阵部分
                # print("rotation matrix:", rotation_matrix)
                # 计算相对旋转矩阵
                rotation_matrix = np.dot(rotation_matrix, np.linalg.inv(base_rotation_matrix))  # 相对旋转矩阵
                # print("invinv      rotation matrix:", rotation_matrix)
                # print("bias_rotation_matrix:", bias_rotation_matrix)
                rotation_matrix = np.dot(rotation_matrix, bias_rotation_matrix)
                # print("relative rotation matrix:", rotation_matrix)
                r = R.from_matrix(rotation_matrix)  # 将旋转矩阵转换为rpy (roll, pitch, yaw)
                # print("rpy:", r)
                rpy = r.as_euler('xyz', degrees=False)  # 使用xyz顺序，角度单位为度
                # print("相对旋转：", rpy)
                # print("22222222222222222222222222222222222222")
                # pose = {
                #     "position": xyz.tolist(),
                #     "orientation": rpy.tolist()
                # }
                pose = xyz.tolist() + rpy.tolist()

                # print("=================prepare pose======================")
                # print("new pose is : ", pose)
                # print("Pose (xyz, rpy):", pose)
                sender.send_target_pose(pose)
                # 接收响应
                response = receiver.receive_response()
                print(f"接收到的数据: {response}")  # 打印接收到的响应
                
                # Sleep to maintain the desired frequency
                elapsed_time = time.time() - start_time
                time_to_sleep = max(0, interval - elapsed_time)
                time.sleep(time_to_sleep)