if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent) #  获取当前文件的父目录的父目录的父目录的路径
    sys.path.append(ROOT_DIR) #  将该路径添加到系统路径中
    os.chdir(ROOT_DIR) #  将当前工作目录更改为该路径

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace): # 从基类BaseWorkspace继承
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)    # inherit from father class __init__ method

        # set seed
        seed = cfg.training.seed    # 在config中的training部分设置的随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)    # create instance of policy class

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)  # 加载ema模型

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())  # 优化器设置，使用config中的参数

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self): 
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDatase # 定义一个BaseImageDatase类型的变量dataset
        dataset = hydra.utils.instantiate(cfg.task.dataset) # 使用hydra.utils.instantiate方法实例化cfg.task.dataset
        assert isinstance(dataset, BaseImageDataset)    # 断言dataset的类型是否BaseImageDataset
        train_dataloader = DataLoader(dataset, **cfg.dataloader)    # 创建训练数据加载器
        normalizer = dataset.get_normalizer()   # 获取数据集的归一化器

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset() # 获取验证集
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader) # 使用配置文件中的参数，创建验证集的数据加载器

        self.model.set_normalizer(normalizer) # 设置模型的归一化器
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=( # 计算训练步骤数，即训练数据集的长度乘以训练的轮数，再除以梯度累积的次数
                len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,  # // 整数除法运算符
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner # 按照config的设置来实例化env_runner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer 把模型和优化器转移到指定的设备上
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        # debug 模式
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt') # 定义日志文件路径，内容包括：train_loss，global_step，epoch，lr
        with JsonLogger(log_path) as json_logger: # 使用JsonLogger类，将日志写入log_path文件
            for local_epoch_idx in range(cfg.training.num_epochs): # 遍历训练的每个epoch
                step_log = dict() # 定义一个字典，用于存储每个epoch的日志
                # ========= train for this epoch ==========
                train_losses = list() # 定义一个列表，用于存储每个batch的loss
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",  # 使用tqdm库，显示训练进度条
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch): # 遍历每个batch
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True)) #  将batch中的数据移动到指定设备上
                        if train_sampling_batch is None: # 如果train_sampling_batch为空，则将batch赋值给它
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch) # 计算原始损失
                        loss = raw_loss / cfg.training.gradient_accumulate_every # 将损失除以梯度累积次数
                        loss.backward() # 反向传播

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step() # 执行优化器的step操作，更新模型参数
                            self.optimizer.zero_grad() # 梯度清零
                            lr_scheduler.step() # 学习率更新
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model) # 更新EMA

                        # logging
                        raw_loss_cpu = raw_loss.item() # 将损失值转换为CPU上的浮点数
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False) # 设置训练损失值
                        train_losses.append(raw_loss_cpu) # 将损失值添加到训练损失列表中
                        step_log = {
                            'train_loss': raw_loss_cpu, # 训练损失值
                            'global_step': self.global_step, # 全局步数
                            'epoch': self.epoch, # 当前轮数
                            'lr': lr_scheduler.get_last_lr()[0] # 当前学习率
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1)) # 判断是否为最后一个batch
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step) # 将训练日志记录到wandb
                            json_logger.log(step_log) # 将训练日志记录到json文件
                            self.global_step += 1 # 全局步数加1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break # 如果达到最大训练步数，则退出循环

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)  
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()   # NOTE：eval是干嘛的？ python Moulde中类属性访问，这里将policy设置为eval模式

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list() # 验证集损失
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))    # lambda是一个匿名函数，这里将batch中的数据移动到指定设备上
                                loss = self.model.compute_loss(batch)   # 计算验证集的损失
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():   # 不进行梯度计算
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action'] # ground_truth action
                        
                        result = policy.predict_action(obs_dict) #  使用策略预测动作
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action) #  计算预测动作和真实动作之间的均方误差
                        step_log['train_action_mse_error'] = mse.item()
                        del batch #  删除batch变量
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint 保存
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()  #设置policy为训练模式

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
