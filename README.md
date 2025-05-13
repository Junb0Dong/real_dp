# real_dp
diffusion policy 部署 
base [diffusion policy](https://github.com/real-stanford/diffusion_policy)

## 环境配置
使用visionpro对机械臂进行遥操作，base [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop)

但对streamer源码进行修改，主要是添加了对visionpro的控制频率，请使用`diffusion_policy/real_world/visionpro_streamer.py`代替[VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop)中的`streamer.py`
