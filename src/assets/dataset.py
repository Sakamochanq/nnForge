import random

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
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

    parts = p.parts
    try:
        sdnet_idx = parts.index("SDNET2018")
        if sdnet_idx + 1 < len(parts):
            structure = parts[sdnet_idx + 1]
            return f"{structure}:{group_id}"
    except ValueError:
        pass

    return group_id


def _grouped_split_indices(
    samples: List[Tuple[str, int]],
    ratios: Tuple[float, float, float],
    seed: int,
):
    if len(ratios) != 3 or not abs(sum(ratios) - 1.0) < 1e-6:
        raise ValueError("ratios must be 3 floats that sum to 1.0")

    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, (path, _class_idx) in enumerate(samples):
        group_to_indices[_sdnet_group_key(path)].append(idx)

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


class DataManager:
    def __init__(self):
        img_size = _get_config_attr("IMG_SIZE", "img_size", default=224)
        
        # 訓練用データの拡張
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            
            # ±15°の範囲でランダムに回転
            transforms.RandomRotation(15),
            
            # 50%で水平反転
            transforms.RandomHorizontalFlip(p=0.5),
            
            # 50%で垂直反転
            transforms.RandomVerticalFlip(p=0.3),
            
            # 明るさ、コントラスト、彩度をランダムに変化させる
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
            
            # アフィン変換（平行移動）
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            
            # ガウシアンブラーを追加
            # ImageNetの平均と標準偏差で正規化
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225]),
        ])
        
        # 検証・テスト用の変換（データ拡張なし）
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

    def load(self):
        data_dir = _get_config_attr("DATA_DIR", "dataset")
        if data_dir is None:
            raise ValueError("Dataset root directory is not set in config (DATA_DIR or dataset)")

        # 訓練用と検証用で異なるデータセットインスタンスを作成
        full_dataset = datasets.ImageFolder(data_dir, transform=self.val_transform)

        seed = int(_get_config_attr("SEED", "seed", default=42))
        train_idx, val_idx, test_idx = _grouped_split_indices(
            full_dataset.samples,
            ratios=(0.70, 0.15, 0.15),
            seed=seed,
        )

        # 訓練データはデータ拡張を適用
        train_dataset_aug = datasets.ImageFolder(data_dir, transform=self.train_transform)
        train_data = Subset(train_dataset_aug, train_idx)
        
        # 検証とテストデータは拡張なし
        val_data = Subset(full_dataset, val_idx)
        test_data = Subset(full_dataset, test_idx)

        batch_size = int(_get_config_attr("BATCH_SIZE", "batch_size", default=32))

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, val_loader, full_dataset.classes