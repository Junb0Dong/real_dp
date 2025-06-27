import numpy as np
from scipy.spatial.transform import Rotation

# 输入四元数 [qw, qx, qy, qz]
quaternion = [0.999956, 0.00941685, 7.72212e-07, -4.60362e-05]

# 使用scipy的Rotation类进行转换
# 注意：scipy期望四元数格式为 [qx, qy, qz, qw]，因此需要调整顺序
rot = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])

# 获取欧拉角（默认为ZYX顺序，即Yaw-Pitch-Roll）
euler_angles_rad = rot.as_euler('ZYX', degrees=False)  # 弧度制
euler_angles_deg = rot.as_euler('ZYX', degrees=True)   # 角度制

# 输出结果
print("欧拉角（弧度制）：")
print(f"Roll (X轴旋转): {euler_angles_rad[2]:.6f} rad")
print(f"Pitch (Y轴旋转): {euler_angles_rad[1]:.6f} rad")
print(f"Yaw (Z轴旋转): {euler_angles_rad[0]:.6f} rad")

print("\n欧拉角（角度制）：")
print(f"Roll (X轴旋转): {euler_angles_deg[2]:.6f}°")
print(f"Pitch (Y轴旋转): {euler_angles_deg[1]:.6f}°")
print(f"Yaw (Z轴旋转): {euler_angles_deg[0]:.6f}°")