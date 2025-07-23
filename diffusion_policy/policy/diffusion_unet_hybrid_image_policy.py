from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    """
    DiffusionUnetHybridImagePolicy 类是一个基于图像的策略类，继承自 BaseImagePolicy。
    该类结合了扩散模型和 U-Net 架构，用于处理图像观测并生成动作预测。
    """
    def __init__(self, 
            shape_meta: dict,   # data的shape_meta，包含了动作和观测的shape，对应config中policy.shape_meta
            noise_scheduler: DDPMScheduler, # 噪声调度器，用于生成噪声，config中policy.noise_scheduler也有相关配置
            horizon,    # 预测的步数，对应config中policy.horizon
            n_action_steps, # 动作步数，对应config中policy.n_action_steps
            n_obs_steps,    # 观测步数，对应config中policy.n_obs_steps
            num_inference_steps=None,   # 推理步骤数
            obs_as_global_cond=True,    # 是否将观测作为全局条件
            # 与cofig不一致（在PushT实验中），但由于使用 hydra 进行配置管理，配置文件中的参数会被视为用户的显式设置，其优先级要高于类 __init__ 方法中的默认参数，会覆盖__init__ 方法中的默认参数。
            crop_shape=(76, 76), 
            diffusion_step_embed_dim=256,   # 扩散嵌入的维度
            down_dims=(256,512,1024),   # 下采样维度
            kernel_size=5,   # 卷积核大小
            n_groups=8,   # 分组数量
            cond_predict_scale=True,    # 是否在条件预测中使用缩放
            obs_encoder_group_norm=False,   # 是否观察编码器组归一化
            eval_fixed_crop=False,  # 是否评估固定裁剪
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta parse:解析
        action_shape = shape_meta['action']['shape']    # 动作形状
        assert len(action_shape) == 1   # 确保动作形状是一维的
        action_dim = action_shape[0]    # 动作维度
        obs_shape_meta = shape_meta['obs']  # 观测形状元数据
        obs_config = {  # 创建观测配置字典
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict() # 创建一个字典，用于存储观察形状
        for key, attr in obs_shape_meta.items(): # 遍历观察形状元数据，key是观察数据的名称，attr是一个包含该观察数据属性的字典
            shape = attr['shape'] # 从 attr 字典中提取 shape 键对应的值，这个值表示当前观察数据的形状
            obs_key_shapes[key] = list(shape) # 将形状转换为列表后存储到obs_key_shapes字典中

            type = attr.get('type', 'low_dim')  # 从 attr 字典中获取 type 键对应的值，如果该键不存在，则默认值为 'low_dim'
            if type == 'rgb':   # 如果当前观察数据的类型是 'rgb'，则将其名称 key 添加到 obs_config 字典的 'rgb' 列表中
                obs_config['rgb'].append(key)   # 观察数据类型为rgb就可以，key的名字可以是其他任意的
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs'] # 获取观测编码器
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0] # 获取obs_encoder的输出形状
        input_dim = action_dim + obs_feature_dim    # 计算输入维度
        global_cond_dim = None # 全局条件维度
        if obs_as_global_cond: # 如果obs_as_global_cond为True，则输入维度为action_dim，全局条件维度为obs_feature_dim * n_obs_steps
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D( #  创建条件Unet1D模型
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model  # Unet模型
        self.noise_scheduler = noise_scheduler #  噪声调度器
        self.mask_generator = LowdimMaskGenerator( #  低维掩码生成器
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        """
        该函数最主要的作用是实现了反向扩散过程（Decoder）
        conditional_sample 函数的主要作用是基于给定的条件数据（condition_data）和条件掩码（condition_mask），通过扩散模型生成符合条件的样本轨迹
        参数：
        self：类的实例对象，用于访问类的属性和方法。
        condition_data：条件数据，作为生成样本的已知信息。
        condition_mask：条件掩码，用于指示 condition_data 中哪些部分是有效的。
        local_cond：局部条件，默认为 None。
        global_cond：全局条件，默认为 None。
        generator：随机数生成器，用于生成随机噪声，默认为 None。
        **kwargs：可变关键字参数，将传递给 scheduler.step 方法。
        """
        model = self.model  # 获取扩散模型
        scheduler = self.noise_scheduler    # 获取噪声调度器

        trajectory = torch.randn( # 生成与condition_data形状相同的随机噪声张量trajectory,作为扩散过程的初始轨迹
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)   # 设置用于生成样本的时间步数

        for t in scheduler.timesteps: # 遍历每个时间步
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask] # 确保去噪过程中的样本满足条件约束

            # 2. predict model output
            model_output = model(trajectory, t,     # 将当前的xxx输入到扩散模型中，得到模型的输出
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(    # 去噪过程，计算上一时刻的trajectory
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # 再次确保生成的样本 trajectory 符合给定的条件，将 condition_data 中有效部分赋值给 trajectory 对应的位置
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        动作预测函数，基于观测字典进行动作预测。
        参数:
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict) # 对观测字典进行归一化处理
        value = next(iter(nobs.values())) # 获取归一化后的观测值
        B, To = value.shape[:2] # 获取观测值的形状，B是批次大小，To是观测时间步长
        T = self.horizon # 获取预测的时间步长，T是时间维度，D是空间维度
        Da = self.action_dim # 获取动作的维度
        Do = self.obs_feature_dim # 获取观测特征的维度
        To = self.n_obs_steps # 获取观测的时间步长

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))   # 将前 To 个时间步的观测数据进行重塑
            nobs_features = self.obs_encoder(this_nobs) # 使用obs编码器将观测数据转换为特征向量
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)  # 重塑特征向量
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(  # 调用 conditional_sample 函数进行采样，得到上一时刻的trajectory
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da] # 提取采样结果中的动作预测部分
        action_pred = self.normalizer['action'].unnormalize(naction_pred)   # 反归一化动作预测结果

        """
        To = 3
        Ta = 4
        T = 6
        |o|o|o|
        | | |a|a|a|a|
        |o|o|
        | |a|a|a|a|a|
        | | | | |a|a|
        """
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        """
        将传入的 normalizer 对象的状态（即其内部的参数）复制到当前策略类实例的 self.normalizer 中
        """
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        数据进行归一化后，对action添加噪声，并使用模型预测噪声残差，计算损失。
        """
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])  # 输入数据进行归一化
        nactions = self.normalizer['action'].normalize(batch['action']) # 对动作进行归一化处理
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions   # 这里的trajectory是归一化后的动作数据
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)  # 使用 self.mask_generator 生成一个与 trajectory 形状相同的掩码，用于指示哪些部分需要进行修复

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device) # 生成一个与 trajectory 形状相同的噪声
        bsz = trajectory.shape[0]   # 获取 batch size
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond) # 对加噪后的trajectory进行预测，得到预测的噪声残差

        pred_type = self.noise_scheduler.config.prediction_type # 获取预测类型，可以是 'epsilon' 或 'sample'
        if pred_type == 'epsilon':
            target = noise  # 如果预测类型是 'epsilon'，则目标值为噪声
        elif pred_type == 'sample':
            target = trajectory # 如果预测类型是 'sample'，则目标值为原始轨迹
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')   # 计算预测值和目标值之间的均方误差损失，reduction='none'表示不对损失进行平均或求和处理
        loss = loss * loss_mask.type(loss.dtype)    # 将损失掩码应用到损失中
        loss = reduce(loss, 'b ... -> b (...)', 'mean') # 将损失进行维度 reduction，并计算平均值
        loss = loss.mean()
        return loss
