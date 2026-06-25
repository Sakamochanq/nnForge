from assets.dataset import DataManager
from assets.model import Model
from assets.train import Train

# データを読み込む（assets/config.py の設定に従う）
data = DataManager()
# classes には抽出用の ["background", "crack"] が返されます
train_loader, val_loader, classes = data.load()

# 抽出用（U-Net）モデルの作成
model = Model().build()

# 学習の実行（内部で BCEWithLogitsLoss や mIoU の計算が行われます）
trainer = Train(model, train_loader, val_loader)
trainer.train()

print("\033[92mAll training completed.\n\033[0m")