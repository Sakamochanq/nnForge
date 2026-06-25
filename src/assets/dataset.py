import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from PIL import Image
import torch

# データ拡張に albumentations を使用します
import albumentations as A
from albumentations.pytorch import ToTensorV2

from assets.config import config


def _get_config_attr(*names, default=None):
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _sdnet_group_key(image_path: str) -> str:
    p = Path(image_path)
    stem = p.stem  # e.g. "7039-110_2"
    group_id = stem.split("-", 1)[0]
    return group_id


def _grouped_split_indices(
    samples: List[Tuple[Path, Path]],
    ratios: Tuple[float, float, float],
    seed: int,
):
    if len(ratios) != 3 or not abs(sum(ratios) - 1.0) < 1e-6:
        raise ValueError("ratios must be 3 floats that sum to 1.0")

    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, (img_path, _) in enumerate(samples):
        group_to_indices[_sdnet_group_key(str(img_path))].append(idx)

    groups = [(g, idxs) for g, idxs in group_to_indices.items()]
    rng = random.Random(seed)
    rng.shuffle(groups)
    
    groups.sort(key=lambda x: len(x[1]), reverse=True)

    total = len(samples)
    targets = [
        int(total * ratios[0]),
        int(total * ratios[1]),
        total - int(total * ratios[0]) - int(total * ratios[1]),
    ]
    current = [0, 0, 0]
    out: List[List[int]] = [[], [], []]  # train, val, test

    def choose_split(n: int) -> int:
        deficits = [targets[i] - current[i] for i in range(3)]
        candidates = [i for i in range(3) if deficits[i] >= n]
        if candidates:
            return max(candidates, key=lambda i: deficits[i])
        overflow = [max(0, current[i] + n - targets[i]) for i in range(3)]
        return min(range(3), key=lambda i: (overflow[i], current[i]))

    for _group, idxs in groups:
        split = choose_split(len(idxs))
        out[split].extend(idxs)
        current[split] += len(idxs)

    return out[0], out[1], out[2]


# --- 抽出用に新たに追加したカスタムDatasetクラス ---
class CrackSegmentationDataset(Dataset):
    """画像とマスク画像のペアを読み込むデータセット"""
    def __init__(self, samples: List[Tuple[Path, Path]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # 画像をRGBで読み込み、numpy配列に変換
        image = np.array(Image.open(img_path).convert("RGB"))
        # マスク（白黒）をグレースケールで読み込み、numpy配列に変換
        mask = np.array(Image.open(mask_path).convert("L"))

        # マスクを0（背景）と1（ひび割れ）に二値化
        mask = (mask > 128).astype(np.float32)

        if self.transform:
            # albumentationsで画像とマスクを「同時に」変形
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # マスクの形状を [H, W] から [1, H, W] に拡張
            mask = mask.unsqueeze(0)

        return image, mask


class DataManager:
    def __init__(self):
        img_size = _get_config_attr("IMG_SIZE", "img_size", default=224)
        
        # 訓練用データの拡張（albumentations版）
        # 画像とマスクが連動して同じ角度・方向に回転/反転します
        self.train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Rotate(limit=15, p=0.5),                  # ±15°の範囲でランダムに回転
            A.HorizontalFlip(p=0.5),                    # 50%で水平反転
            A.VerticalFlip(p=0.3),                      # 30%で垂直反転
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5), # 色調変化 (画像のみに適用される)
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet正規化
            ToTensorV2()                                # PyTorchのテンソルに変換
        ])
        
        # 検証・テスト用の変換（データ拡張なし）
        self.val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    def load(self):
        data_dir = _get_config_attr("DATA_DIR", "dataset")
        if data_dir is None:
            raise ValueError("Dataset root directory is not set in config (DATA_DIR or dataset)")

        data_path = Path(data_dir)
        images_dir = data_path / "images"
        masks_dir = data_path / "masks"

        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(f"ディレクトリ構成が正しくありません。{images_dir} と {masks_dir} が必要です。")

        # 画像の一覧を取得し、対応するマスクのパスとペアにする
        # (拡張子はデータセットに合わせて .jpg などを指定、大文字小文字対応)
        samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(images_dir.glob("*.*")):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                # 画像と同じファイル名がマスクフォルダにある前提
                mask_path = masks_dir / img_path.name
                if mask_path.exists():
                    samples.append((img_path, mask_path))

        if len(samples) == 0:
            raise ValueError(f"有効な画像とマスクのペアが見つかりませんでした。パスを確認してください: {images_dir}")

        seed = int(_get_config_attr("SEED", "seed", default=42))
        train_idx, val_idx, test_idx = _grouped_split_indices(
            samples,
            ratios=(0.70, 0.15, 0.15),
            seed=seed,
        )

        # 訓練・検証・テストのデータセットを構築
        full_samples = samples
        train_samples = [full_samples[i] for i in train_idx]
        val_samples = [full_samples[i] for i in val_idx]
        test_samples = [full_samples[i] for i in test_idx]

        train_data = CrackSegmentationDataset(train_samples, transform=self.train_transform)
        val_data = CrackSegmentationDataset(val_samples, transform=self.val_transform)
        test_data = CrackSegmentationDataset(test_samples, transform=self.val_transform)

        batch_size = int(_get_config_attr("BATCH_SIZE", "batch_size", default=32))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        # 抽出タスクではクラス一覧（full_dataset.classes）の代わりに
        # タスク識別のためのダミーリスト、またはテストデータを返すように調整します
        return train_loader, val_loader, ["background", "crack"]