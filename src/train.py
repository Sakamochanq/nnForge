import torch
from ultralytics import YOLO


# デバイスの設定（GPU or CPU）
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
if __name__ == '__main__':
    
    # v8m-cls は分類用標準モデル（自動ダウンロード）
    model = YOLO('yolov8m-cls.pt')
    
    print(f"Using device: {device}")

    results = model.train(
        data='./dataset/images/',
        epochs=150,
        imgsz=640,
        device=device
    )