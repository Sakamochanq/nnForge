import torch.nn as nn
from torchvision import models


class Model:
    def build(self):
        
        # 既存の学習済みモデル ResNet18 を使用する
        model = models.resnet18(weights="DEFAULT")
        
        # 既存の学習済みモデルの重みを固定する
        # つまり、ResNet18の特徴抽出部分は学習させず、最終層のみを学習させる
        for param in model.parameters():
            param.requires_grad = False
            
        # レイヤー3層目の追加
        for param in model.layer3.parameters():
            param.requires_grad = True
        
        # レイヤー4層目の追加
        for param in model.layer4.parameters():
            param.requires_grad = True

        # 最終層を2クラス分類用に変更
        # model.fc = nn.Linear(model.fc.in_features, 2)
        
        model.fc = nn.Sequential(nn.Dropout(p=0.5),  nn.Linear(model.fc.in_features, 2))

        return model