import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from torchvision import transforms
from assets.config import config


class Predict:
    def __init__(self, model, classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(config.model, map_location=self.device))

        self.model.eval()

        self.classes = classes

        self.transform = transforms.Compose([transforms.Resize((config.img_size, config.img_size)), transforms.ToTensor()])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)

        # 勾配を計算しない
        with torch.no_grad():
            # 予測
            output = self.model(image)
            # 確率の計算
            probabilities = torch.softmax(output, dim=1)
            _, pred = torch.max(probabilities, 1)

        result = self.classes[pred.item()]
        
        
        print(f"\nPredicted: {result}\n")
        for idx, class_name in enumerate(self.classes):
            percentage = probabilities[0][idx].item() * 100
            print(f"{class_name}: {percentage:.2f}%")
        print("")



class GradCAM:
    def __init__(self, model, classes, device):
        self.model = model
        self.classes = classes
        self.device = device
        
        # 特徴マップを保存する変数
        self.feature_maps = None
        # 特徴マップの勾配を保存する変数
        self.gradients = None
        
        # ResNet18の最後の畳み込み層（layer4）にフックを設定
        # フックは、フォワードパスでレイヤーの出力を保存する
        self.hook_forward = model.layer4.register_forward_hook(self.hook_fn_forward)
        # バックワードパスでレイヤーの勾配を保存する
        self.hook_backward = model.layer4.register_full_backward_hook(self.hook_fn_backward)

    # 特徴マップを保存
    def hook_fn_forward(self, module, input, output):
        self.feature_maps = output.detach()

    # 勾配を保存
    def hook_fn_backward(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_gradcam(self, image_path, target_class_idx=None):
        
        # 画像を読み込む
        image = Image.open(image_path).convert("RGB")
        original_image = np.array(image)
        
        # 画像を前処理
        transform = transforms.Compose([transforms.Resize((config.img_size, config.img_size)), transforms.ToTensor()])
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # 勾配を計算するため、require_grad_を有効化
        image_tensor.requires_grad = True
        
        # モデルのフォワードパス
        output = self.model(image_tensor)
        
        # ターゲットクラスを指定しない場合、予測クラスを使用
        if target_class_idx is None:
            target_class_idx = torch.argmax(output, dim=1).item()
        
        # ターゲットクラスのスコアを取得
        target_score = output[0, target_class_idx]
        
        # バックワードパスで勾配を計算
        self.model.zero_grad()
        target_score.backward()
        
        # チャネルごとの重要度を計算
        # gradients の形状: (1, 512, 7, 7)
        # 各チャネルの勾配を平均化して重みを計算
        weights = torch.mean(self.gradients, dim=(2, 3))[0, :]
        
        # 重み付けされた特徴マップを結合
        # feature_maps の形状: (1, 512, 7, 7)
        cam = torch.zeros(self.feature_maps.shape[2], self.feature_maps.shape[3]).to(self.device)
        
        for i in range(weights.shape[0]):
            cam += weights[i] * self.feature_maps[0, i, :, :]
        
        # 負の値を除去（ReLU）
        cam = torch.relu(cam)
        
        # 0～1に正規化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # NumPyに変換
        cam = cam.cpu().detach().numpy()
        
        # 元の画像サイズにリサイズ
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # ヒートマップを生成（JETカラーマップを使用）
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 元の画像とヒートマップを重ねる（60%の透明度）
        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        return overlay, cam_resized, self.classes[target_class_idx]

    # フックを削除
    def remove_hooks(self):
        self.hook_forward.remove()
        self.hook_backward.remove()