from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

'''
概述：PushTImageDataset 类是一个基于图像数据集的类，用于处理和采样图像数据。它继承自 BaseImageDataset 类，并使用 ReplayBuffer 来存储和访问数据。该类支持从给定的 Zarr 文件路径加载图像数据，并提供了方法来采样数据并将其转换为 PyTorch 张量。

参数：
zarr_path：字符串，Zarr 文件的路径，用于加载图像数据。
horizon：整数，默认为 1，表示采样的序列长度。
pad_before：整数，默认为 0，表示在序列开始前填充的帧数。
pad_after：整数，默认为 0，表示在序列结束后填充的帧数。
seed：整数，默认为 42，用于随机数生成器的种子。
val_ratio：浮点数，默认为 0.0，表示验证集的比例。
max_train_episodes：整数，默认为 None，表示训练集的最大集数。

返回值：
__getitem__ 方法返回一个字典，包含以下键值对：
'obs'：包含观测数据的字典，包括 'image'（图像数据，形状为 (T, 3, 96, 96)）和 'agent_pos'（代理位置数据，形状为 (T, 2)）。
'action'：动作数据，形状为 (T, 2)，并转换为 PyTorch 张量。

示例：
dataset = PushTImageDataset(zarr_path='path/to/zarr/file.zarr')
sample_data = dataset[0]  # 获取索引为 0 的样本数据
'''
class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,  # zarr 文件路径
            horizon=1,  # 采样序列长度
            pad_before=0,   # 序列开始前填充的帧数
            pad_after=0,    # 序列结束后填充的帧数
            seed=42,    # 随机种子
            val_ratio=0.0,  # 验证集比例
            max_train_episodes=None # 最大训练集集数，如果为 None 则不进行下采样
            ):
        """
        加载zarr数据，划分训练集和验证集，并创建序列采样器。
        """
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action']) # 加载image和state-action pair数据
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)  # 验证集掩码
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)  # 对训练集进行下采样

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)    # 创建一个序列采样器，用于从回放缓冲区中采样序列数据
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        """
        创建验证集数据集，使用与训练集相同的采样器，但使用验证集掩码。
        返回一个新的 PushTImageDataset 实例，包含验证集数据。
        """
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )   # 创建一个验证集采样器
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """
        为state-action pair数据和图像数据创建一个归一化器。
        参数 mode 指定归一化的模式，kwargs 可用于传递其他参数。
        返回一个 LinearNormalizer 实例，包含归一化器和图像数据的归一化器。
        """
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }   # state-action pair数据
        normalizer = LinearNormalizer() # 创建一个线性归一化器
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)   # 拟合归一化器，针对action和agent_pos数据
        normalizer['image'] = get_image_range_normalizer()  # 为图像数据添加归一化器
        return normalizer

    def __len__(self) -> int:
        """
        返回采样器的长度，即数据集中可采样的序列数量。
        """
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        将采样得到的数据转换为标准的数据格式。
        参数：
        sample：一个字典，包含采样得到的图像、状态和动作数据。
        """
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255 # 图像数据的通道维度移动到第二维，并将像素值归一化到 [0, 1] 范围

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }   # data字典，包含观测数据和动作数据
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取指定索引的样本数据，并将其转换为 PyTorch 张量。
        """
        sample = self.sampler.sample_sequence(idx)  # 从采样器中获取样本数据
        data = self._sample_to_data(sample) # 将样本数据转换为标准格式
        torch_data = dict_apply(data, torch.from_numpy) # 将数据转换为 PyTorch 张量
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
