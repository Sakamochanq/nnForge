from pathlib import Path
from assets.config import config
from assets.model import Model
from assets.predict import Predict


def main():
    print("="*50)
    print("推論スクリプト")
    print("="*50 + "\n")
    
    # モデルパスの入力
    model_path = input("モデルパスを入力してください: ").strip()
    
    if not model_path:
        model_path = config.model
        print(f"デフォルトを使用: {model_path}")
    
    if not Path(model_path).exists():
        print(f"エラー: モデルファイルが見つかりません - {model_path}")
        return
    
    # config.modelを一時的に変更（Predictクラスが使用するため）
    original_model_path = config.model
    config.model = model_path
    
    # 画像パスの入力
    image_path = input("画像パスを入力してください: ").strip()
    
    if not image_path:
        print("エラー: 画像パスが入力されていません")
        config.model = original_model_path
        return
    
    if not Path(image_path).exists():
        print(f"エラー: 画像ファイルが見つかりません - {image_path}")
        config.model = original_model_path
        return
    
    # 推論の実行
    print("\n推論を実行中...\n")
    
    try:
        # モデルの構築
        model = Model().build()
        
        # クラス情報（デフォルト）
        classes = ["class_0", "class_1"]
        
        # 推論
        predictor = Predict(model, classes)
        predictor.predict(image_path)
        
        print("="*50)
        print("推論完了")
        print("="*50)
    except Exception as e:
        print(f"推論中にエラーが発生しました: {e}")
    finally:
        # config.modelを元に戻す
        config.model = original_model_path


if __name__ == "__main__":
    main()
