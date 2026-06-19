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

        # 最適化アルゴリズム ｜ 今回は ADAM を使用
        # 損失関数（どれだけ間違ったか）
        self.criterion = nn.CrossEntropyLoss()
        
        self.optim = optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-4) # weight_decay は重み減衰。過学習を防ぐための正則化手法（0.0001を適用）
        
        # 学習率スケジューラーの初期化
        self.scheduler = lr_scheduler.create(self.optim)
        
        # 学習曲線用のhistory
        self.train_acc_rec = []
        self.val_acc_rec = []
        self.train_loss_rec = []
        self.val_loss_rec = []
        self.lr_rec = []

        # Early Stopping 用の変数
        self._es_counter      = 0             # 改善なし連続エポック数
        self._es_best_acc     = float('-inf') # これまでの最高 val_acc
        self._es_best_weights = None          # ベスト時の重み（CPUコピー）

    # 学習
    def train(self):
        stopped_early = False  # Early Stopping で中断したかどうか

        for epoch in range(config.epochs):
            # 学習モード ON
            self.model.train()

            # 正解枚数
            correct = 0
            
            # 総枚数
            total = 0
            
            # 損失の合計（間違いの合計）
            loss_sum = 0
            
            # 進捗表示
            print('')
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", unit="batch", colour="cyan")
            
            for images, labels in loop:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 勾配降下法の計算を正しく行うため、モデル内のパラメータの勾配を初期化
                # 勾配降下法は、機械学習モデルの誤差（損失）を最小にするパラメータ（重み）を見つけるための反復最適化手法
                # https://www.ibm.com/jp-ja/think/topics/gradient-descent
                self.optim.zero_grad()

                # 予測の出力
                outputs = self.model(images)
                
                # 出力と正解ラベルの損失を計算                
                loss = self.criterion(outputs, labels)
                
                # 勾配を計算
                loss.backward()
                self.optim.step()

                # 損失の加算
                loss_sum = loss_sum + loss.item()

                # 予測の最大値を取得
                _, pred = torch.max(outputs, 1)
                
                # 総枚数の加算
                total += labels.size(0)
                
                # 正解枚数の加算
                correct += (pred == labels).sum().item()
                
                # 更新
                loop.set_postfix(loss=f"{loss_sum / (total / config.batch_size):.4f}", accuracy=f"{100 * correct / total:.4f}%")

                # 正解率の計算（正解枚数 / 総枚数）
                train_acc = 100 * correct / total
            
            # 検証の正解率の計算
            val_acc, val_loss = self.validate()
            
            # 学習曲線用にメモリに記録（エポック全体の平均損失）
            avg_train_loss = loss_sum / len(self.train_loader)
            self.train_acc_rec.append(train_acc)
            self.val_acc_rec.append(val_acc)
            self.train_loss_rec.append(avg_train_loss)
            self.val_loss_rec.append(val_loss)
            
            # 現在の学習率を記録
            current_lr = self.optim.param_groups[0]['lr']
            self.lr_rec.append(current_lr)
            
            # スケジューラーのステップ
            lr_scheduler.step(self.scheduler, val_acc)

            # 結果の出力
            print("\033[96m" + f"学習回数 {epoch+1}/{config.epochs} | " f"訓練損失 {avg_train_loss:.4f} | 検証損失 {val_loss:.4f} | " f"学習正解率 {train_acc:.4f}% | " f"検証正解率 {val_acc:.4f}% \n" + "\033[0m")

            # Early Stopping チェック（改善なしが続いたら中断）
            if self.early_stopping(val_acc):
                stopped_early = True
                break

        # 早期終了時はファイル名に earlystop を付加
        suffix = "-es" if stopped_early else ""

        # 学習モデルの保存
        # sate_dict()でモデルの重みを保存
        model_path = f"{config.model_dir}Model-{config.epochs}-{config.batch_size}-{config.learning_rate}{suffix}.pth"
        torch.save(self.model.state_dict(), model_path)

        #出力する文字を緑にして
        print(f"Model Saved " + "\033[92m" + "Successfully" + "\033[0m \n")
        
        # 学習曲線の可視化
        self.learning_curve(suffix)


    # 検証
    def validate(self):
        
        # 検証モード ON
        self.model.eval()

        correct = 0
        total = 0
        val_loss_sum = 0

        # 勾配を計算しない
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss_sum += loss.item()
                
                _, pred = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
        # 検証の正解率の計算（正解枚数 / 総枚数）
        val_acc = 100 * correct / total
        # エポック全体の平均検証損失
        avg_val_loss = val_loss_sum / len(self.val_loader)
        return val_acc, avg_val_loss


    # Early Stopping
    def early_stopping(self, val_acc: float,
                        patience: int = 10, min_delta: float = 0.0) -> bool:
        
        if val_acc > self._es_best_acc + min_delta:
            
            self._es_best_acc     = val_acc
            self._es_counter      = 0
            self._es_best_weights = copy.deepcopy(self.model.state_dict())
            print(f"\033[92m  [es] best val_acc → {val_acc:.4f}%\033[0m")
            return False

        # 改善なし：カウントアップ
        self._es_counter += 1
        print(f"\033[93m  [es] no improvement ({self._es_counter}/{patience})\033[0m")

        if self._es_counter >= patience:
            # patience 超過 → ベスト重みを復元して終了シグナルを返す
            if self._es_best_weights is not None:
                self.model.load_state_dict(self._es_best_weights)
                print(f"\033[91m  [es] stopped. Best weights restored "
                      f"(val_acc {self._es_best_acc:.4f}%)\033[0m\n")
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
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
            
        # ./Curve-<epoch>-<batch_size>-<learning_rate>.png として保存 🐧
        pingu = f'Curve-{config.epochs}-{config.batch_size}-{config.learning_rate}{suffix}.png'
        
        plt.savefig(pingu, dpi=300, bbox_inches='tight')
        print(f"{pingu} saved " + "\033[92m" + "Successfully" + "\033[0m \n")
        plt.close()