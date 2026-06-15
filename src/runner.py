# クラスのインポート
from xml.parsers.expat import model
from assets.config import config
from assets.dataset import DataManager
from assets.model import Model
from assets.predict import Predict

import torch


# データを読み込む
data = DataManager()
train_loader, val_loader, classes = data.load()
    
# モデルの作成
model = Model().build()

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load(config.model, map_location=device))
model.eval()
    
# Predictorの定義
predictor = Predict(model, classes, device)

print("Classes:", classes)

# 任意の画像から推論
root = input('\n Root ❯ ').strip()

# 通常の予測を実行
predictor.predict_all(root)