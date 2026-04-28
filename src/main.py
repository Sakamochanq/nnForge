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


print("\033[92m\nAll training completed.\n\033[0m")
input()

# 予測
# predictor = Predict(model, classes)
# predictor.predict("test.jpg") 