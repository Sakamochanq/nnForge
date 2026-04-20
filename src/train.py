import torch

from ultralytics import YOLO


# デバイスの設定（GPU or CPU）
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


if __name__ == '__main__':
    
    # v26m-cls は分類用標準モデル
    model = YOLO('yolov26m-cls.pt')

    results = model.train(
        data="./dataset/images/train",
        epochs=150,
        imgsz=640,
        device=device
    )