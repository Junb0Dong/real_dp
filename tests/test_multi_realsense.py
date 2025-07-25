import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import json
import time
import numpy as np
from diffusion_policy.real_world.multi_realsense import MultiRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder

def test():
    config = json.load(open('/home/junbo/diffusion_policy/diffusion_policy/real_world/realsense_config/435_high_accuracy_mode.json', 'r'))

    # 图像缩放函数
    def transform(data):
        color = data['color']
        h,w,_ = color.shape
        factor = 4
        color = cv2.resize(color, (w//factor,h//factor), interpolation=cv2.INTER_AREA)
        # color = color[:,140:500]
        data['color'] = color
        return data

    from diffusion_policy.common.cv2_util import get_image_transform
    color_transform = get_image_transform(
        input_res=(640,480),
        output_res=(320,240), 
        bgr_to_rgb=False)
    def transform(data):
        data['color'] = color_transform(data['color'])
        return data

    # one thread per camera
    video_recorder = VideoRecorder.create_h264(
        fps=30,
        codec='h264',
        thread_type='FRAME'
    )
    
    with MultiRealsense(
            # resolution=(1280,720),
            resolution=(640,480),
            # resolution=(1280,240),
            capture_fps=15,
            record_fps=15,
            enable_color=True,
            # advanced_mode_config=config,
            transform=transform,
            # recording_transform=transform,
            # video_recorder=video_recorder,
            verbose=True
        ) as realsense:
        print("the num of cameras: ", realsense.n_cameras)
        realsense.set_exposure(exposure=150, gain=5)
        intr = realsense.get_intrinsics()
        print(intr)
        

        video_path = 'data/realsense'
        rec_start_time = time.time() + 1
        realsense.start_recording(video_path, start_time=rec_start_time)
        realsense.restart_put(rec_start_time)

        out = None
        vis_img = None
        while True:
            out = realsense.get(out=out)
            print("out:", type(out), out)

            # 获取所有相机的 color 图像
            bgr_list = [out[k]['color'] for k in sorted(out.keys())]
            vis_img = np.concatenate(bgr_list, axis=0)  # 垂直拼接
            # vis_img = np.concatenate(bgr_list, axis=1)  # 水平拼接

            cv2.imshow('multi_realsense', vis_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1/60)
            if time.time() > (rec_start_time + 20.0):
                break


if __name__ == "__main__":
    test()
