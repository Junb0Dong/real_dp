import multiprocessing as mp
import numpy as np
import time
from avp_stream import VisionProStreamer
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

class VisionPro(mp.Process):
    def __init__(self, 
            shm_manager, 
            get_max_k=30, 
            frequency=30,
            dtype=np.float32,
            visionpro_ip = "10.15.202.91",
            ):
        """
        Continuously listen to VisionPro events
        and update the latest state.
        """
        super().__init__()

        # copied variables
        self.frequency = frequency
        self.dtype = dtype
        self.timestamps = []    # 记录时间戳
        example = {
            # 假设 VisionPro 返回的信息包含 left_wrist 等数据
            'left_wrist': np.zeros((4, 4), dtype=dtype),  # left_wrist 4x4 matrix
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

        # VisionPro 设备相关
        self.avp_ip = visionpro_ip  # 可以作为参数传入
        self.visionpro_streamer = VisionProStreamer(ip=self.avp_ip, record=True)

    # ======= get state APIs ==========

    def get_left_wrist_state(self):
        state = self.ring_buffer.get()
        print(f"Retrieved left_wrist state: {state['left_wrist']}")
        return np.array(state['left_wrist'], dtype=self.dtype)

    # 在这里还可以加获取其他状态的函数
    
    #========== start stop API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop ==========
    def run(self):
        try:
            # send one message immediately so client can start reading
            left_wrist = np.zeros((4, 4), dtype=self.dtype)
            self.ring_buffer.put({
                'left_wrist': left_wrist,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()

            while not self.stop_event.is_set():
                receive_timestamp = time.time()
                try:
                    r = self.visionpro_streamer.latest
                    left_wrist = r['left_wrist']
                    state = {
                        'left_wrist': np.array(left_wrist, dtype=self.dtype),
                        'receive_timestamp': receive_timestamp
                    }
                    time.sleep(0.01)
                    self.ring_buffer.put(state)
                    self.timestamps.append(receive_timestamp)  # 
                    print(f"Received left_wrist data: {left_wrist}")
                    print(f"timestamps: {self.timestamps}")
                except Exception as e:
                    print(f"Error getting VisionPro data: {e}")
                time.sleep(1/self.frequency)
        finally:
            pass

    def get_frequency(self):
        if len(self.timestamps) < 2:
            return 0
        intervals = np.diff(self.timestamps)
        average_interval = np.mean(intervals)
        frequency = 1 / average_interval if average_interval > 0 else 0
        return frequency