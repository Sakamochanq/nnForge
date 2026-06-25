import torch
from assets.config import config
from assets.dataset import DataManager
from assets.model import Model
from assets.predict import Predict

# データの読み込み（クラスリスト ["background", "crack"] を取得するため）
data = DataManager()
_, _, classes = data.load()
    
# 抽出用（U-Net）モデルの構築
model = Model().build()

# デバイスの設定（GPUが使えればGPU、なければCPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# config.py で指定された学習済みモデルの重み（.pth）をロード
# ※ config.model パスが正しいことを確認してください
model.load_state_dict(torch.load(config.model, map_location=device))
model.eval()
    
# 抽出用 Predictor の定義
predictor = Predict(model, classes, device)

print("Classes:", classes)

# 推論したいデータセットのルートディレクトリを入力
# 例: dataset/SDNET2018 (直下に images/ と masks/ がある構成)
root = input('\n Root ❯ ').strip()

# 抽出用の予測を実行（予測された赤色の半透明重ね合わせ画像が output_predictions/ に自動保存されます）
predictor.predict_all(root)