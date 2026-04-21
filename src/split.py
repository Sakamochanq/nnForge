import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 出力ディレクトリが存在しなければ作成
    output_path.mkdir(parents=True, exist_ok=True)
    
    # train, val, test フォルダを作成
    for split in ['train', 'val', 'test']:
        split_path = output_path / split
        split_path.mkdir(exist_ok=True)
        
        # クラスフォルダを作成
        for class_name in os.listdir(source_path):
            class_path = split_path / class_name
            class_path.mkdir(exist_ok=True)
    
    # 各クラスごとに分割
    for class_name in os.listdir(source_path):
        class_source_path = source_path / class_name
        
        # クラスフォルダ内の画像を取得
        if not class_source_path.is_dir():
            continue
        
        images = [f for f in os.listdir(class_source_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        
        print(f"クラス '{class_name}': {len(images)} 枚のデータセット")
        
        # train と val+test に分割
        train_images, temp_images = train_test_split(
            images,
            train_size=train_ratio,
            random_state=42
        )
        
        # val と test に分割
        val_size = val_ratio / (val_ratio + test_ratio)
        val_images, test_images = train_test_split(
            temp_images,
            train_size=val_size,
            random_state=42
        )
        
        print(f"  → train: {len(train_images)}, val: {len(val_images)}, test: {len(test_images)}")
        
        # ファイルをコピー
        for img_name in train_images:
            src = class_source_path / img_name
            dst = output_path / 'train' / class_name / img_name
            shutil.copy2(src, dst)
        
        for img_name in val_images:
            src = class_source_path / img_name
            dst = output_path / 'val' / class_name / img_name
            shutil.copy2(src, dst)
        
        for img_name in test_images:
            src = class_source_path / img_name
            dst = output_path / 'test' / class_name / img_name
            shutil.copy2(src, dst)


if __name__ == '__main__':
    # 元のデータセット場所)
    source = input("Dataset Origin ❯ ")
    
    # 出力先
    output = './dataset/images'
    
    print("データセット分割を開始します...")
    print(f"元のデータ: {source}")
    print(f"出力先: {output}")
    print()
    
    split_dataset(source, output, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    print("\n分割完了!")
    print(f"✓ {output}/train")
    print(f"✓ {output}/val")
    print(f"✓ {output}/test")