import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from PIL import Image
from torchvision import transforms
from assets.config import config


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class Predict:
    def __init__(self, model, classes, device):
        """
        runner.py 側でセットアップ済みの model / device をそのまま受け取る。
        load_state_dict・eval() は呼び出し元で済ませておくこと。
        """
        self.device = device
        self.model = model
        self.classes = classes

        self.transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor()
        ])

    def _predict_single(self, image_path):
        """1枚の画像を推論し、(予測クラス名, 確率リスト) を返す"""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            prob = torch.softmax(output, dim=1)
            _, pred = torch.max(prob, 1)

        pred_class = self.classes[pred.item()]
        prob_list = [prob[0][i].item() for i in range(len(self.classes))]
        return pred_class, prob_list

    def _collect_images(self, folder_path):
        """フォルダ内の対応画像ファイルパスを収集する"""
        images = []
        for fname in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                images.append(os.path.join(folder_path, fname))
        return images

    def predict_folder(self, folder_path, label=None):
        """
        1つのフォルダを一括推論する。
        label: 正解ラベル（Crack / UnCrack など）。指定すると正解率を計算。
        """
        images = self._collect_images(folder_path)
        if not images:
            print(f"  [警告] 画像が見つかりませんでした: {folder_path}\n")
            return []

        results = []
        correct = 0

        print(f"\n{'─'*55}")
        print(f"  フォルダ: {folder_path}  ({len(images)} 枚)")
        if label:
            print(f"  正解ラベル: {label}")
        print(f"{'─'*55}")

        for img_path in images:
            fname = os.path.basename(img_path)
            pred_class, prob_list = self._predict_single(img_path)

            is_correct = (pred_class == label) if label else None
            if is_correct:
                correct += 1

            mark = ""
            if is_correct is True:
                mark = " ✓"
            elif is_correct is False:
                mark = " ✗"

            prob_str = "  |  ".join(
                f"{self.classes[i]}: {prob_list[i]*100:.2f}%"
                for i in range(len(self.classes))
            )
            print(f"  [{fname}]")
            print(f"    Predicted: {pred_class}{mark}")
            print(f"    {prob_str}")

            results.append({
                "path": img_path,
                "predicted": pred_class,
                "probs": prob_list,
                "correct": is_correct,
            })

        # フォルダ集計
        print(f"{'─'*55}")
        if label:
            acc = correct / len(images) * 100
            print(f"  集計: {correct}/{len(images)} 正解  ({acc:.1f}%)")
        print(f"{'─'*55}\n")

        return results

    def predict_all(self, data_root):
        """
        data_root/Crack と data_root/UnCrack を一括推論し、全体の集計を表示する。
        """
        all_results = []
        total_correct = 0
        total_count = 0

        print(f"\n{'='*55}")
        print(f"  Predict  |  root: {data_root}")
        print(f"{'='*55}")

        # data_root直下のサブフォルダを正解ラベルとして処理
        subfolders = sorted([
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ])

        if not subfolders:
            print(f"[エラー] サブフォルダが見つかりませんでした: {data_root}")
            return

        for subfolder in subfolders:
            folder_path = os.path.join(data_root, subfolder)
            # サブフォルダ名を正解ラベルとして使用（classes に含まれる場合）
            label = subfolder if subfolder in self.classes else None
            results = self.predict_folder(folder_path, label=label)

            all_results.extend(results)
            total_count += len(results)
            total_correct += sum(1 for r in results if r.get("correct"))

        # 全体集計
        if total_count > 0:
            overall_acc = total_correct / total_count * 100
            print(f"\n{'='*55}")
            print(f"  全体集計")
            print(f"{'='*55}")
            print(f"  総画像数  : {total_count} 枚")
            print(f"  正解数    : {total_correct} 枚")
            print(f"  正解率    : {overall_acc:.2f}%")

            # クラスごとの正解率
            print(f"\n  クラス別 正解率:")
            for cls in self.classes:
                cls_results = [r for r in all_results if r.get("correct") is not None
                               and r["path"].split(os.sep)[-2] == cls]
                if cls_results:
                    cls_correct = sum(1 for r in cls_results if r["correct"])
                    cls_acc = cls_correct / len(cls_results) * 100
                    print(f"    {cls:>10}: {cls_correct}/{len(cls_results)}  ({cls_acc:.1f}%)")
            print(f"{'='*55}\n")

            # 混同行列
            self._print_confusion_matrix(all_results)
            self._plot_confusion_matrix(all_results)

    def _build_confusion_matrix(self, all_results):
        """
        正解ラベルが取得できた結果のみを対象に混同行列を構築する。
        返り値: cm[true_idx][pred_idx] の2次元リスト
        """
        n = len(self.classes)
        cm = [[0] * n for _ in range(n)]
        cls_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for r in all_results:
            true_label = os.path.basename(os.path.dirname(r["path"]))
            if true_label not in cls_to_idx:
                continue
            true_idx = cls_to_idx[true_label]
            pred_idx = cls_to_idx[r["predicted"]]
            cm[true_idx][pred_idx] += 1

        return cm

    def _print_confusion_matrix(self, all_results):
        """混同行列をコンソールにテキスト表示する"""
        cm = self._build_confusion_matrix(all_results)
        n = len(self.classes)
        col_w = max(len(c) for c in self.classes) + 2

        print(f"\n{'='*55}")
        print(f"  混同行列 (行: 正解ラベル / 列: 予測ラベル)")
        print(f"{'='*55}")

        # ヘッダー行
        header = " " * (col_w + 2) + "".join(f"{c:^{col_w}}" for c in self.classes)
        print(header)
        print(" " * (col_w + 2) + "─" * (col_w * n))

        for i, true_cls in enumerate(self.classes):
            row = f"  {true_cls:<{col_w}}"
            for j in range(n):
                row += f"{cm[i][j]:^{col_w}}"
            print(row)

        # 各クラスの Precision / Recall / F1
        print(f"\n  {'クラス':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'─'*44}")
        for i, cls in enumerate(self.classes):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(n)) - tp
            fn = sum(cm[i][j] for j in range(n)) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            print(f"  {cls:<12} {precision:>9.2%} {recall:>10.2%} {f1:>10.2%}")

        print(f"{'='*55}\n")

    def _plot_confusion_matrix(self, all_results, save_path=f"Matrix-{config.epochs}-{config.batch_size}-{config.learning_rate}.png"):
        """混同行列をヒートマップとして保存する"""
        cm = self._build_confusion_matrix(all_results)
        n = len(self.classes)
        cm_array = np.array(cm)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(self.classes, fontsize=12)
        ax.set_yticklabels(self.classes, fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=13)
        ax.set_ylabel("True Label", fontsize=13)
        ax.set_title("Confusion Matrix", fontsize=14, pad=12)

        # セルに数値を表示
        thresh = cm_array.max() / 2.0
        for i in range(n):
            for j in range(n):
                color = "white" if cm_array[i, j] > thresh else "black"
                ax.text(j, i, str(cm_array[i, j]),
                        ha="center", va="center",
                        fontsize=15, fontweight="bold", color=color)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
        print(f"  混同行列を保存しました: {save_path}\n")