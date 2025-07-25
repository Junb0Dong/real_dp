"""
备份修改的 VisionProStreamer 类
因为对源代码添加了频率控制，所以备份一下，后面重装环境后需要覆盖源代码的这个文件。
路径参考：/home/junbo/anaconda3/envs/robodiff/lib/python3.9/site-packages/avp_stream/streamer.py
"""

import grpc
from avp_stream.grpc_msg import * 
from threading import Thread
from avp_stream.utils.grpc_utils import * 
import time 
import numpy as np 
from collections import deque 

YUP2ZUP = np.array([[[1, 0, 0, 0], 
                    [0, 0, -1, 0], 
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]]], dtype = np.float64)


class VisionProStreamer:

    def __init__(self, ip, record = True, frequency = 30): 

        # Vision Pro IP 
        self.ip = ip
        self.record = record 
        self.recording = [] 
        self.latest = None 
        self.axis_transform = YUP2ZUP
        self.frequency = frequency
        self.last_update_time = 0
        self.period_buffer = deque(maxlen=10)  # 移动平均窗口 [[7]]
        self.start_streaming()



    def start_streaming(self): 

        stream_thread = Thread(target = self.stream)
        stream_thread.start() 
        while self.latest is None: 
            pass 
        print(' == DATA IS FLOWING IN! ==')
        print('Ready to start streaming.') 


    def stream(self): 

        request = handtracking_pb2.HandUpdate()
        try:
            with grpc.insecure_channel(f"{self.ip}:12345") as channel:
                stub = handtracking_pb2_grpc.HandTrackingServiceStub(channel)
                responses = stub.StreamHandUpdates(request)
                for response in responses:
                    current_time = time.time()
                    # print(f"Received new data at {time.time()}")
                    delta = current_time - self.last_update_time
                    if delta >= 1/self.frequency:
                        self.period_buffer.append(delta)
                        transformations = {
                            "left_wrist": self.axis_transform @  process_matrix(response.left_hand.wristMatrix),
                            "right_wrist": self.axis_transform @  process_matrix(response.right_hand.wristMatrix),
                            "left_fingers":   process_matrices(response.left_hand.skeleton.jointMatrices),
                            "right_fingers":  process_matrices(response.right_hand.skeleton.jointMatrices),
                            "head": rotate_head(self.axis_transform @  process_matrix(response.Head)) , 
                            "left_pinch_distance": get_pinch_distance(response.left_hand.skeleton.jointMatrices),
                            "right_pinch_distance": get_pinch_distance(response.right_hand.skeleton.jointMatrices),
                            # "rgb": response.rgb, # TODO: should figure out how to get the rgb image from vision pro 
                        }
                        # raw_matrix = process_matrix(response.left_hand.wristMatrix)
                        # print(f"Raw matrix from process_matrix: {raw_matrix}")
                        transformations["right_wrist_roll"] = get_wrist_roll(transformations["right_wrist"])
                        transformations["left_wrist_roll"] = get_wrist_roll(transformations["left_wrist"])
                        # print(f"Transformed left_wrist: {transformations['left_wrist']}")
                        self.last_update_time = current_time
                        # print(f"数据更新于 {current_time}")
                        if self.record: 
                            self.recording.append(transformations)
                        self.latest = transformations 

        except Exception as e:
            print(f"An error occurred: {e}")
            pass 

    def get_latest(self): 
        return self.latest
        
    def get_recording(self): 
        return self.recording
    
    # 滑动平均法计算频率
    def get_real_frequency(self): 
        weights = [1/(i+1) for i in range(len(self.period_buffer))]
        weighted_sum = sum(d * w for d, w in zip(self.period_buffer, weights))
        avg_period = weighted_sum / sum(weights)
        # 应用相位差补偿算法 [[4]]
        if len(self.period_buffer) >= 5:
            phase_comp = (self.period_buffer[-1] - self.period_buffer[0]) / len(self.period_buffer)
            avg_period -= phase_comp
        return 1 / avg_period if avg_period > 0 else 0.0
    
if __name__ == "__main__": 

    streamer = VisionProStreamer(ip = '10.29.230.57')
    while True: 

        latest = streamer.get_latest()
        # print(latest)