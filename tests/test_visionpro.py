"""

# 原测试代码
from avp_stream import VisionProStreamer
avp_ip = "10.15.202.91"   # example IP 
s = VisionProStreamer(ip = avp_ip, record = True, frequency=1)

while True:
    print("Waiting for data...")
    r = s.latest
    print("data:", r['left_wrist'])
    print("frequency is :", s.get_real_frequency())

"""

import threading
import queue
import numpy as np
from avp_stream import VisionProStreamer
import time

avp_ip = "10.15.202.91"
s = VisionProStreamer(ip=avp_ip, record=True, frequency=10)

data_queue = queue.Queue()

def data_collector():
    last_data = None
    while True:
        current_data = s.latest
        try:
            # 使用np.array_equal进行数组比较 [[3]]
            if last_data is None or not np.array_equal(current_data, last_data):
                data_queue.put(current_data.copy())  # 避免引用问题 [[1]]
                last_data = current_data.copy()
        except Exception as e:
            print(f"Data comparison error: {e}")
        
        # 动态调整采样间隔保持频率同步
        time.sleep(1.0 / max(s.get_real_frequency(), 1))  # 防止除零错误

# 设置守护线程确保主程序退出时自动终止
collector_thread = threading.Thread(target=data_collector, daemon=True)
collector_thread.start()

try:
    while True:
        data = data_queue.get()
        print("New data received:")
        print("Left wrist:\n", data['left_wrist'])
        print("Real frequency:", s.get_real_frequency())
except KeyboardInterrupt:
    print("Exiting program...")