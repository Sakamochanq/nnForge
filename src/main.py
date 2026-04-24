from assets.dataset import DataManager
from assets.model import Model
from assets.train import Train
from assets.predict import Predict

# データを読み込む（config.py）
data = DataManager()
train_loader, val_loader, classes = data.load()

# モデルの作成
model = Model().build()

# 学習
trainer = Train(model, train_loader, val_loader)
trainer.train()

# 予測
predict = Predict(model, classes)
predict.predict("test.jpg")