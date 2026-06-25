import copy

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from assets.config import config
from assets.lr_scheduler import lr_scheduler
from tqdm import tqdm


class Train:
    # 初期化
    def __init__(self, model, train_loader, val_loader):
        
        # GPUが利用可能であれば使用する。それ以外はCPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 使用デバイスの確認
        print(f"\033[92m\nUsing device: {self.device}\033[0m")
        
        # GPU/CPUにモデルを送る
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 抽出（セグメンテーション）用の損失関数に変更
        # ピクセル単位での2値分類を行うため、BCEWithLogitsLossを使用します
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 最適化アルゴリズム ｜ 今回は ADAM を使用
        self.optim = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        
        # 学習率スケジューラーの初期化
        self.scheduler = lr_scheduler.create(self.optim)
        
        # 学習曲線用のhistory (acc_rec から iou_rec に名称変更)
        self.train_iou_rec = []
        self.val_iou_rec = []
        self.train_loss_rec = []
        self.val_loss_rec = []
        self.lr_rec = []

        # Early Stopping 用の変数
        self._es_counter      = 0            # 改善なし連続エポック数
        self._es_best_loss    = float('inf') # これまでの最小 val_loss
        self._es_best_weights = None         # ベスト時の重み（CPUコピー）

    # --- 抽出タスク用の評価指標 (IoU) 計算関数 ---
    def _calculate_iou(self, outputs, masks, threshold=0.5):
        """ピクセル単位でIoU（ジャッカード係数）を計算するヘルパー関数"""
        # シグモイド関数で0〜1の確率に変換し、閾値で0か1の2値マスクにする
        preds = (torch.sigmoid(outputs) > threshold).float()
        
        # 積集合（共通部分）と和集合の計算
        intersection = (preds * masks).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
        
        # 分母が0になる（どちらも完全に背景）場合のゼロ除算対策
        eps = 1e-6
        iou = (intersection + eps) / (union + eps)
        
        # バッチ全体の平均値を返す
        return iou.mean().item()

    # 学習
    def train(self):
        stopped_early = False  # Early Stopping で中断したかどうか

        for epoch in range(config.epochs):
            # 学習モード ON
            self.model.train()
            
            # 損失の合計とIoUの合計
            loss_sum = 0
            iou_sum = 0
            total_batches = 0
            
            # 進捗表示
            print('')
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", unit="batch", colour="cyan")
            
            # セグメンテーションでは labels の代わりに masks (画像ラベル) を受け取る
            for images, masks in loop:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # 勾配の初期化
                self.optim.zero_grad()

                # 予測の出力 [バッチサイズ, 1, H, W]
                outputs = self.model(images)
                
                # 出力と正解マスクの損失を計算                
                loss = self.criterion(outputs, masks)
                
                # 勾配を計算
                loss.backward()
                self.optim.step()

                # 損失の加算
                loss_sum += loss.item()
                
                # バッチごとのIoUを計算して加算
                batch_iou = self._calculate_iou(outputs, masks)
                iou_sum += batch_iou
                total_batches += 1
                
                # 進捗バーの更新 (Accuracyの代わりにmIoUを表示)
                current_avg_loss = loss_sum / total_batches
                current_avg_iou = iou_sum / total_batches
                loop.set_postfix(loss=f"{current_avg_loss:.4f}", mIoU=f"{current_avg_iou * 100:.2f}%")

            # 訓練データのエポック平均指標
            train_loss = loss_sum / len(self.train_loader)
            train_iou = (iou_sum / len(self.train_loader)) * 100
            
            # 検証の評価（IoUと損失の計算）
            val_iou, val_loss = self.validate()
            
            # 学習曲線用に履歴に記録
            self.train_iou_rec.append(train_iou)
            self.val_iou_rec.append(val_iou)
            self.train_loss_rec.append(train_loss)
            self.val_loss_rec.append(val_loss)
            
            # 現在の学習率を記録
            current_lr = self.optim.param_groups[0]['lr']
            self.lr_rec.append(current_lr)
            
            # スケジューラーのステップ (指標として val_iou を渡す、または設定次第で val_loss)
            lr_scheduler.step(self.scheduler, val_iou)

            # 結果の出力
            print("\033[96m" + f"学習回数 {epoch+1}/{config.epochs} | " 
                  f"訓練損失 {train_loss:.4f} | 検証損失 {val_loss:.4f} | " 
                  f"訓練mIoU {train_iou:.2f}% | 検証mIoU {val_iou:.2f}% \n" + "\033[0m")

            # Early Stopping チェック（検証損失の改善なしが続いたら中断）
            if self.early_stopping(val_loss):
                stopped_early = True
                break

        # 早期終了時などのファイル保存プレフィックス/サフィックス処理
        suffix = "" if stopped_early else ""

        # 学習モデルの保存
        model_path = f"{config.model_dir}Model-{config.epochs}-{config.batch_size}-{config.learning_rate}{suffix}.pth"
        torch.save(self.model.state_dict(), model_path)

        print(f"Model Saved " + "\033[92m" + "Successfully" + "\033[0m \n")
        
        # 学習曲線の可視化 (内部のプロットもiou用に修正してください)
        self.learning_curve(suffix)

    # 検証
    def validate(self):
        # 検証モード ON
        self.model.eval()

        val_loss_sum = 0
        val_iou_sum = 0

        # 勾配を計算しない
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss_sum += loss.item()
                
                # 検証データのIoU計算
                val_iou_sum += self._calculate_iou(outputs, masks)
                
        # エポック全体の平均検証指標の計算
        avg_val_loss = val_loss_sum / len(self.val_loader)
        avg_val_iou = (val_iou_sum / len(self.val_loader)) * 100
        return avg_val_iou, avg_val_loss

    # Early Stopping (元のロジックを100%完全維持)
    def early_stopping(self, val_loss: float, patience: int = 5, min_delta: float = 0.0) -> bool:
        if val_loss < self._es_best_loss - min_delta:
            self._es_best_loss    = val_loss
            self._es_counter      = 0
            self._es_best_weights = copy.deepcopy(self.model.state_dict())
            print(f"\033[94m  [es] 損失率 減少 {val_loss:.4f}\033[0m")
            return False

        self._es_counter += 1
        print(f"\033[91m  [es] 損失率 増加 ({self._es_counter}/{patience})\033[0m")

        if self._es_counter >= patience:
            if self._es_best_weights is not None:
                self.model.load_state_dict(self._es_best_weights)
                print(f"\033[91m  [es] stopped. Best weights restored "
                      f"(val_loss {self._es_best_loss:.4f})\033[0m\n")
            return True

        return False



    # 学習曲線の描画
    def learning_curve(self, suffix: str = ""):
        
        epochs = range(1, len(self.train_acc_rec) + 1)
        
        plt.figure(figsize=(16, 5))
        
        # 損失の描画
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_loss_rec, marker='o', label='Train', linewidth=2)
        plt.plot(epochs, self.val_loss_rec, marker='s', label='Valid', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 正解率の描画
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_acc_rec, marker='o', label='Train', linewidth=2)
        plt.plot(epochs, self.val_acc_rec, marker='s', label='Valid', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 学習率の推移を描画
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.lr_rec, marker='D', label='Learning Rate', linewidth=2, color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('lr Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
            
        # ./Curve-<epoch>-<batch_size>-<learning_rate>.png として保存 🐧
        pingu = f'Curve-{config.epochs}-{config.batch_size}-{config.learning_rate}{suffix}.png'
        
        plt.savefig(pingu, dpi=300, bbox_inches='tight')
        print(f"{pingu} saved " + "\033[92m" + "Successfully" + "\033[0m \n")
        plt.close()