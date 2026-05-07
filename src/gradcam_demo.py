"""
Grad-CAM可視化のデモンストレーション
CNNが注目した領域をヒートマップで表示します
"""

import os
from assets.model import Model
from assets.predict import Predict


def main():
    # クラス名の定義
    classes = ["Cracked", "Undamaged"]  # 状態に合わせて変更してください
    
    # モデルの作成
    model = Model().build()
    
    # Predictクラスの初期化（Grad-CAMも自動で初期化）
    predictor = Predict(model, classes)
    
    # テスト画像のパスを指定
    # dataset/images/SDNET2018/W/CW または /UW フォルダから選択
    test_image_path = input("Image : ") # 実際のパスに変更してください
    
    # 画像ファイルが存在するかチェック
    if not os.path.exists(test_image_path):
        print(f"Error: Image file not found at {test_image_path}")
        print("Please provide a valid image path from the dataset.")
        return
    
    # Grad-CAM付きで予測を実施
    # save_path パラメータで可視化結果を保存できます
    result, probabilities, overlay_image = predictor.predict_with_gradcam(
        test_image_path,
        save_path="gradcam_result.png"  # 結果を画像ファイルに保存
    )
    
    print("Grad-CAM visualization completed!")


if __name__ == "__main__":
    main()
