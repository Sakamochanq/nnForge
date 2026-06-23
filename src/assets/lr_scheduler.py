import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, 
    ExponentialLR, 
    CosineAnnealingLR, 
    ReduceLROnPlateau,

    # ウォームアップ用
    LinearLR,
    
    # 合成用
    SequentialLR
)

from assets.config import config


class lr_scheduler:
    
    @staticmethod
    def create(optim):
        if config.type == "StepLR":
            main_scheduler = StepLR(optim, step_size=config.step_size, gamma=config.gamma)
        
        elif config.type == "ExponentialLR":
            main_scheduler = ExponentialLR(optim, gamma=config.gamma)
        
        elif config.type == "CosineAnnealingLR":
            warmup_epochs = 5
            main_epochs = max(1, config.epochs - warmup_epochs)
            main_scheduler = CosineAnnealingLR(optim, T_max=main_epochs)
        
        elif config.type == "ReduceLROnPlateau":
            return ReduceLROnPlateau(optim, mode='min', factor=config.gamma, patience=5)
        
        else:
            # Bonk!
            print(f"\033[93mBonk!\033[0m")
            return None
        
        # ウォームアップの追加
        warmup_epochs = getattr(config, 'warmup_epochs', 5)
        
        # 本来の学習率の 0.1倍 から 1.0倍 まで直線的に上げる
        warmup_scheduler = LinearLR(optim, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        
        # ウォームアップとメインのスケジューラーを合成させる
        combined_scheduler = SequentialLR(optim, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
        
        return combined_scheduler
    
    @staticmethod
    def step(scheduler, val_acc=None):
        if isinstance(scheduler, ReduceLROnPlateau):
            if val_acc is None:
                raise ValueError("Val_acc Not Found")
            scheduler.step(val_acc)
        else:
            scheduler.step()
