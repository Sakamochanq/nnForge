import torch.nn as nn
# 事前準備で紹介した segmentation-models-pytorch を使用します
import segmentation_models_pytorch as smp


class Model:
    def build(self):
        # 1. バックボーン（特徴抽出器）に ResNet18 を使用した U-Net を構築
        # classes=1 は「ひび割れ確率（0〜1）」をピクセルごとに出力するためです
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )
        
        # 2. 一旦、エンコーダ（ResNet18部分）のすべての重みを固定する
        for param in model.encoder.parameters():
            param.requires_grad = False
            
        # 3. 元のコードの意図に合わせて、深い層（layer3, layer4）だけを解凍（学習対象に）する
        # ※smpライブラリでは内部の層の名前が model.encoder.layer3 になります
        for param in model.encoder.layer3.parameters():
            param.requires_grad = True
            
        for param in model.encoder.layer4.parameters():
            param.requires_grad = True

        # 4. 画像を元のサイズに拡大する「デコーダ層」と「最終出力層」は、
        # 新しく作られる層なので、必ずすべての重みを学習対象（True）にします
        for param in model.decoder.parameters():
            param.requires_grad = True
            
        for param in model.segmentation_head.parameters():
            param.requires_grad = True

        # 元のコードにあった分類用の全結合層（model.fc）は、
        # セグメンテーションモデルには存在しないため削除（または上書き不要）となります。
        # 代わりに、過学習を防ぐためのDropoutを最終層（segmentation_head）の直前に
        # 差し込みたい場合は、以下のように記述できます。
        model.segmentation_head = nn.Sequential(
            nn.Dropout2d(p=0.5), # 画像空間用の2次元ドロップアウト
            model.segmentation_head
        )

        return model