import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from assets.config import config


class GradCAM:
    """Grad-CAMを使用してCNNの注目領域を可視化"""
    
    def __init__(self, model, target_layer, device):
        self.model = model
        self.device = device
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # フックを登録
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """フォワードパスの活性化を保存"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """バックワードパスの勾配を保存"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image_tensor):
        """Grad-CAMヒートマップを生成"""
        # モデルの出力を取得
        output = self.model(image_tensor)
        
        # 予測クラスを取得
        pred_class = output.argmax(dim=1)
        
        # 逆伝播
        self.model.zero_grad()
        output[0, pred_class].backward()
        
        # 勾配と活性化を取得
        gradients = self.gradients
        activations = self.activations
        
        # チャネルごとに勾配を平均
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # 重み付き活性化マップを計算
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # ヒートマップを正規化 (0-1の範囲)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam, pred_class.item()
    
    def overlay_heatmap(self, image, cam, alpha=0.5):
        """ヒートマップを元画像に重ねる"""
        # CAMをリサイズして元の画像サイズに合わせる
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # ヒートマップにカラーマップを適用
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 元画像とヒートマップを重ね合わせ
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay, heatmap, cam_resized


class Predict:
    def __init__(self, model, classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(config.model, map_location=self.device))

        self.model.eval()

        self.classes = classes
        
        # Grad-CAM用のターゲットレイヤーを設定 (ResNet18の最後の残差ブロック)
        self.gradcam = GradCAM(self.model, self.model.layer4[1].conv2, self.device)

        self.transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor()
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # 勾配を計算しない
        with torch.no_grad():
            # 予測
            output = self.model(image_tensor)
            # 確率の計算
            probabilities = torch.softmax(output, dim=1)
            _, pred = torch.max(probabilities, 1)

        result = self.classes[pred.item()]
        
        
        print(f"\nPredicted: {result}\n")
        for idx, class_name in enumerate(self.classes):
            percentage = probabilities[0][idx].item() * 100
            print(f"{class_name}: {percentage:.2f}%")
        print("")
    
    def predict_with_gradcam(self, image_path, save_path=None):
        """Grad-CAMヒートマップ付きで予測を実施"""
        # 元画像を読み込む
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # PIL画像として読み込み
        image_pil = Image.open(image_path).convert("RGB")
        
        # 変換を適用
        image_tensor = self.transform(image_pil).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad = True
        
        # Grad-CAMを計算
        cam, pred_class = self.gradcam.generate_cam(image_tensor)
        
        # CAMを NumPy配列に変換
        cam_numpy = cam[0, 0].cpu().numpy()
        
        # ヒートマップを元画像に重ねる
        overlay, heatmap, cam_resized = self.gradcam.overlay_heatmap(original_image, cam_numpy)
        
        # 予測結果を取得
        output = self.model(image_tensor.detach())
        probabilities = torch.softmax(output, dim=1)
        result = self.classes[pred_class]
        
        # 結果を表示
        print(f"\n=== Grad-CAM Prediction ===")
        print(f"Predicted: {result}\n")
        for idx, class_name in enumerate(self.classes):
            percentage = probabilities[0][idx].item() * 100
            print(f"{class_name}: {percentage:.2f}%")
        print("")
        
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 元画像
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Grad-CAMヒートマップ
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        
        # 重ね合わせ画像
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (Predicted: {result})")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return result, probabilities[0].cpu().detach().numpy(), overlay
