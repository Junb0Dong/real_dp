"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

# import os
# os.environ["WANDB_API_KEY"] = "0412b933c6b562fef5f1356dcb63e4f5e98f194f"
# os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_DISABLED"] = "true"

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# config path: default(diffusion_policy/config)
# config name: need to be specified in the command line
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config')) 
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)  # resolve the config, interpolate the values(${}), and return the resolved config

    cls = hydra.utils.get_class(cfg._target_)   # import _target_ class from .yaml file, get class
    workspace: BaseWorkspace = cls(cfg) # 实例化从配置中加载的类，并传入完整配置对象
    workspace.run() # run the instance

if __name__ == "__main__":
    main()
