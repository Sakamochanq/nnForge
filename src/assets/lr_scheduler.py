import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

from assets.config import config


class lr_scheduler:
    
    # 各習率スケジューラーごとに条件分岐（設定）
    @staticmethod
    def create(optim):
        if config.type == "StepLR":
            return StepLR(optim, step_size=config.step_size, gamma=config.gamma)
        
        elif config.type == "ExponentialLR":
            return ExponentialLR(optim, gamma=config.gamma)
        
        elif config.type == "CosineAnnealingLR":
            return CosineAnnealingLR(optim, T_max=config.epochs)
        
        elif config.type == "ReduceLROnPlateau":
            return ReduceLROnPlateau(optim, mode='max', factor=config.gamma, patience=5, verbose=True)
        
        else:
            # Bonk!
            print(f"\033[93mBonk!\033[0m")
            return None
    
    @staticmethod
    def step(scheduler, val_acc=None):
        
        if config.type == "ReduceLROnPlateau":
            if val_acc is None:
                raise ValueError("Val_acc Not Found")
            scheduler.step(val_acc)
        else:
            scheduler.step()


