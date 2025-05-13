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
                 dtype=np.float64,
                 visionpro_ip="10.15.202.91"):
        """
        Continuously listen to VisionPro events and update the latest state.

        Args:
            shm_manager: Shared memory manager for ring buffer.
            get_max_k: Maximum attempts to get data from ring buffer (default: 30).
            frequency: Desired update frequency in Hz (default: 30).
            dtype: Data type for numpy arrays (default: np.float32).
            visionpro_ip: IP address of the VisionPro device (default: "10.15.202.91").
        """
        super().__init__()

        # Configuration variables
        self.frequency = frequency
        self.dtype = dtype
        self.timestamps = []  # To track timestamps for frequency calculation
        self.avp_ip = visionpro_ip

        # Example data structure for ring buffer
        example = {
            'left_wrist': np.zeros((4, 4), dtype=dtype),  # 4x4 matrix for left wrist pose
            'right_wrist': np.zeros((4, 4), dtype=dtype),  # 4x4 matrix for right wrist pose
            'receive_timestamp': time.time()
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # Synchronization events
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()



    # ======= Get State APIs ==========

    def get_left_wrist_state(self):
        """
        Retrieve the latest left wrist state from the ring buffer.
        """
        state = self.ring_buffer.get()
        # print(f"Retrieved state from ring_buffer: {state['left_wrist']}")
        return np.array(state['left_wrist'], dtype=self.dtype)
    
    def get_right_wrist_state(self):
        """
        Retrieve the latest left wrist state from the ring buffer.
        """
        state = self.ring_buffer.get()
        # print(f"Retrieved state from ring_buffer: {state['left_wrist']}")
        return np.array(state['right_wrist'], dtype=self.dtype)

    # ======= Start/Stop APIs ==========

    def start(self, wait=True):
        """Start the process and optionally wait for it to be ready."""
        super().start()
        if wait:
            self.ready_event.wait()

    def stop(self, wait=True):
        """Stop the process and optionally wait for it to terminate."""
        self.stop_event.set()
        if wait:
            self.join()

    def __enter__(self):
        """Context manager entry point."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.stop()

    # ======= Main Loop ==========

    def run(self):
        """Main loop to continuously update the shared memory with VisionPro data."""
        try:
            # VisionPro streamer
            self.visionpro_streamer = VisionProStreamer(ip=self.avp_ip, record=True, frequency=self.frequency)
            # Initial state to allow immediate client reading
            left_wrist = np.zeros((4, 4), dtype=self.dtype)
            right_wrist = np.zeros((4, 4), dtype=self.dtype)
            self.ring_buffer.put({
                'left_wrist': left_wrist,
                'right_wrist': right_wrist,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()

            while not self.stop_event.is_set():
                receive_timestamp = time.time()
                try:
                    # Get the latest data from VisionPro streamer
                    r = self.visionpro_streamer.latest
                    # print("r is :", r['left_wrist'], "\n")
                    # print("frequency is :", self.visionpro_streamer.get_real_frequency())
                    if r is None:
                        print("Warning: No data received yet from   VisionProStreamer")
                        continue
                    left_wrist = r['left_wrist']
                    right_wrist = r['right_wrist']
                    state = {
                        'left_wrist': np.array(left_wrist, dtype=self.dtype),
                        'right_wrist': np.array(right_wrist, dtype=self.dtype),
                        'receive_timestamp': receive_timestamp
                    }

                    self.ring_buffer.put(state)
                    self.timestamps.append(receive_timestamp)
                except Exception as e:
                    print(f"Error getting VisionPro data: {e}")
                time.sleep(0.5/self.frequency)  # Control update rate
        finally:
            pass  # Add cleanup if necessary (e.g., closing streamer)