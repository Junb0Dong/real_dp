"""
visionpro_shard_memory.py的单元测试
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


if __name__ == '__main__':
    print("Running VisionPro test...")
    with SharedMemoryManager() as shm_manager:
        with VisionPro(shm_manager=shm_manager, frequency=10, dtype=np.float64, visionpro_ip="10.12.174.92") as vp:
            time.sleep(1.0)
            
            print("VisionPro started.")
            while True:
                retrieved_data = vp.get_right_wrist_state()  # 获取左手腕状态
                print("get data", retrieved_data)
                time.sleep(0.1)