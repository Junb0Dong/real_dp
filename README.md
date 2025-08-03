# real_dp
diffusion policy 部署 
base [diffusion policy](https://github.com/real-stanford/diffusion_policy)

## Installation
使用visionpro对机械臂进行遥操作，base [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop)

但对streamer源码进行修改，主要是添加了对visionpro的控制频率，请使用`diffusion_policy/real_world/visionpro_streamer.py`代替[VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop)中的`streamer.py`

### HardWare 
- Realman Robot X1
- VisionPro X1
- Realsense D435 X2
- gripper X1

### SoftWare
- Ubuntu 20.04
- [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop)
- Realsense SDK
- Diffusion Policy

## Train and Eval on Realman Robot
### Train

使用visionpro来遥操作机械臂采集数据。在遥操作的过程中，Press "C" to start recording. Press "S" to stop recording. Press "Q" to quit，同时确保鼠标在realsense的画面内。
```bash
python realman_real_world.py -o <demo_save_dir> --robot_ip <ip_of_robot> --vp_ip <ip_of_visionpro>
```

保存数据后，将真实的数据转换为`.zarr`格式的数据
```bash
python diffusion_policy/real_world/realman_gripper_env.py -i <demo_collect_dir> -o <demo_train_dir>
```

训练模型，使用`nohup`可以在后台运行
```bash
nohup python train.py --config-name=realman_workspace > train_output.log 2>&1&
```

使用命令来终止训练
```bash
ps aux | grep "python train.py --config-name=realman_workspace" | grep -v grep 
```

### Evaluate
训练好模型后，使用以下命令来评估模型。在eval的过程中，Press "C" to start episode. Press "S" to stop episode. Press "Q" to quit，同时确保鼠标在realsense的画面内。
```bash
python eval_realman_test.py -i <your_checkpiont_dir> -o <ouput_dir>
```

## 部署过程中会出现的问题
- 把本地收集的数据传到服务器上
  ```bash
  scp -r <demo_collect_dir> <username>@<client_ip>:<path>
  ```
- wandb 网络超时，network error
  添加wandb的[镜像源](https://bandw.top/)：
  ```bash
  export WANDB_BASE_URL=https://api.bandw.top
  ```
