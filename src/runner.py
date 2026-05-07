# クラスのインポート
from xml.parsers.expat import model
from assets.dataset import DataManager
from assets.model import Model
from assets.predict import Predict


# データを読み込む
data = DataManager()
train_loader, val_loader, classes = data.load()
    
# モデルの作成
model = Model().build()
    
# Predictorの定義
predictor = Predict(model, classes)
    
# 任意の画像から推論
image = input('\n ❯ ')
predictor.predict(image)