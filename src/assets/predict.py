import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from assets.config import config

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class Predict:
    def __init__(self, model, classes, device):
        """
        runner.py 側でセットアップ済みの model / device をそのまま受け取る。
        """
        self.device = device
        self.model = model
        self.classes = classes  # 抽出では ["background", "crack"] を想定

        # 推論用の基本変換（学習時と同じサイズにリサイズ）
        self.transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _predict_single(self, image_path, threshold=0.5):
        """1枚の画像を推論し、2値化された予測マスク（numpy配列）を返す"""
        image_pil = Image.open(image_path).convert("RGB")
        orig_size = image_pil.size  # (幅, 高さ) を保持

        # テンソル変換してモデルへ入力
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            # シグモイド関数でピクセルごとのひび割れ確率（0〜1）に変換
            prob = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()

        # 確率が閾値を超えたピクセルを1（ひび割れ）とする
        pred_mask = (prob > threshold).astype(np.uint8)

        # 予測結果を元の画像サイズにリサイズして戻す
        pred_mask_resized = np.array(
            Image.fromarray(pred_mask * 255).resize(orig_size, resample=Image.NEAREST)
        ) // 255

        return image_pil, pred_mask_resized

    def _calculate_metrics(self, pred_mask, true_mask):
        """1枚の画像に対するピクセル単位の IoU と Dice 係数を計算"""
        pred = (pred_mask > 0).astype(np.float32)
        true = (true_mask > 0).astype(np.float32)

        intersection = np.sum(pred * true)
        union = np.sum(pred) + np.sum(true) - intersection

        eps = 1e-6
        iou = (intersection + eps) / (union + eps)
        dice = (2.0 * intersection + eps) / (np.sum(pred) + np.sum(true) + eps)

        return iou, dice

    def _save_overlay_result(self, image_pil, pred_mask, save_path):
        """元の画像の上に予測マスクを赤色の半透明で重ね合わせて保存する"""
        img_np = np.array(image_pil)
        overlay = img_np.copy()

        # ひび割れと予測されたピクセル（1）を赤色（RGB: 255, 0, 0）に染める
        overlay[pred_mask == 1] = [255, 0, 0]

        # 元の画像と赤色マスクを 7:3 の割合でブレンド
        blended = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0) if 'cv2' in globals() else None
        
        # OpenCVがない場合の代替処理（PILを使用）
        if blended is None:
            mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8))
            red_image = Image.new("RGB", image_pil.size, (255, 0, 0))
            blended_pil = Image.blend(image_pil, red_image, alpha=0.3)
            # ひび割れの部分だけブレンド画像を適用
            final_image = Image.composite(blended_pil, image_pil, mask_pil)
        else:
            final_image = Image.fromarray(blended)

        # 保存先ディレクトリがなければ作成
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        final_image.save(save_path)

    def _collect_images(self, folder_path):
        """フォルダ内の対応画像ファイルパスを収集する"""
        images = []
        for fname in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                images.append(os.path.join(folder_path, fname))
        return images

    def predict_folder(self, folder_path, masks_folder=None, output_root="output_predictions"):
        """
        1つのフォルダを一括推論する。
        masks_folder: 正解マスク画像があるフォルダのパス。指定するとmIoUを計算。
        """
        images = self._collect_images(folder_path)
        if not images:
            print(f"  [警告] 画像が見つかりませんでした: {folder_path}\n")
            return []

        results = []
        ious = []
        dices = []

        print(f"\n{'─'*55}")
        print(f"  フォルダ: {folder_path}  ({len(images)} 枚)")
        if masks_folder:
            print(f"  正解マスク参照元: {masks_folder}")
        print(f"{'─'*55}")

        for img_path in images:
            fname = os.path.basename(img_path)
            # 推論実行
            image_pil, pred_mask = self._predict_single(img_path)

            # 結果画像の保存パスを決定
            save_path = os.path.join(output_root, os.path.basename(folder_path), f"output_{fname}")
            self._save_overlay_result(image_pil, pred_mask, save_path)

            # 正解データ（マスク画像）が存在する場合、精度指標を計算
            iou, dice = None, None
            if masks_folder:
                mask_path = os.path.join(masks_folder, fname)
                if os.path.exists(mask_path):
                    true_mask = np.array(Image.open(mask_path).convert("L"))
                    true_mask = (true_mask > 128).astype(np.uint8)
                    iou, dice = self._calculate_metrics(pred_mask, true_mask)
                    ious.append(iou)
                    dices.append(dice)

            metric_str = f"  |  IoU: {iou*100:.2f}%  |  Dice: {dice*100:.2f}%" if iou is not None else ""
            print(f"  [{fname}] -> 保存先: {save_path}{metric_str}")

            results.append({
                "path": img_path,
                "iou": iou,
                "dice": dice
            })

        # フォルダ集計
        print(f"{'─'*55}")
        if ious:
            print(f"  フォルダ平均 mIoU: {np.mean(ious)*100:.2f}%")
            print(f"  フォルダ平均 Dice: {np.mean(dices)*100:.2f}%")
        print(f"  ※ 重ね合わせ画像が '{output_root}' に保存されました")
        print(f"{'─'*55}\n")

        return results

    def predict_all(self, data_root, output_root="output_predictions"):
        """
        抽出用のデータセット構成（images / masks）を一括推論する。
        """
        print(f"\n{'='*55}")
        print(f"  Predict (Segmentation)  |  root: {data_root}")
        print(f"{'='*55}")

        images_dir = os.path.join(data_root, "images")
        masks_dir = os.path.join(data_root, "masks")

        # masksフォルダが存在すれば精度を計算する
        has_masks = os.path.exists(masks_dir)
        masks_path = masks_dir if has_masks else None

        results = self.predict_folder(images_dir, masks_folder=masks_path, output_root=output_root)

        # 全体集計
        valid_ious = [r["iou"] for r in results if r["iou"] is not None]
        if valid_ious:
            valid_dices = [r["dice"] for r in results if r["dice"] is not None]
            print(f"\n{'='*55}")
            print(f"  全体集計 (Segmentation)")
            print(f"{'='*55}")
            print(f"  総処理画像数 : {len(results)} 枚")
            print(f"  全体平均 mIoU: {np.mean(valid_ious)*100:.2f}%")
            print(f"  全体平均 Dice: {np.mean(valid_dices)*100:.2f}%")
            print(f"{'='*55}\n")