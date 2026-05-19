# クラスのインポート
from xml.parsers.expat import model
from assets.config import config
from assets.dataset import DataManager
from assets.model import Model
from assets.predict import Predict
from assets.predict import GradCAM

import torch
import cv2
import matplotlib.pyplot as plt


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
predictor = Predict(model, classes)

# Grad-CAM の定義
gradcam = GradCAM(model, classes, device)

# 任意の画像から推論
image = input('\n Image ❯ ')

# 通常の予測を実行
predictor.predict(image)

# Grad-CAM を表示
use_gradcam = input("\033[96mSend Grad-CAM ? (y/n) ❯ \033[0m")
print("")

if use_gradcam.lower() == 'y':
    # Grad-CAM画像を生成
    overlay, _, predicted_class = gradcam.generate_gradcam(image)
    
    # BGR→RGBに変換（OpenCVはBGR)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 7))
    # 元の画像
    plt.subplot(1, 2, 1)
    original_img = cv2.imread(image)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    plt.imshow(original_img)
    plt.axis('off')
    plt.title(image.split("\\")[-1])
    
    # Grad-CAM画像
    plt.subplot(1, 2, 2)
    plt.imshow(overlay_rgb)
    plt.title(f"Result: {predicted_class}")
    plt.axis('off')
    plt.tight_layout()
    
    plt.show()
    
    # matplotlibで表示
    # plt.figure(figsize=(5, 5))
    # plt.imshow(overlay_rgb)
    # plt.title(image.split("\\")[-1])
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()
    
else:
    print("\033[94mGrad-CAM visualization skipped.\033[0m")
    print("")