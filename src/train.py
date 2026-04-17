import torch

from ultralytics import YOLO


# デバイスの設定（GPU or CPU）
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


if __name__ == '__main__':
    
    # -cls は分類用モデル
    model = YOLO('yolov8n-cls.pt')

    results = model.train(
        data="./dataset/images/train",
        epochs=100,
        imgsz=640,
        device=device
    )