import torch.nn as nn
from torchvision import models


class Model:
    def build(self):
        
        # 既存の学習済みモデル ResNet18 を使用する
        model = models.resnet18(weights="DEFAULT")
        
        # 既存の学習済みモデルの重みを固定する
        # for param in model.parameters():
        #     param.requires_grad = False

        # 最終層を2クラス分類用に変更
        model.fc = nn.Linear(model.fc.in_features, 2)

        return model