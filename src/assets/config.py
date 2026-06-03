class config:
    
    # 学習させるデータセット
    dataset = "D:\\Enviroments\\nnForge\\src\\dataset\\images\\SDNET2018\\W";
    
    #画像サイズ
    img_size = 224;
    
    # バッチサイズ
    batch_size = 32;
    
    # 学習回数
    epochs = 100;
    
    # 学習率
    learning_rate = 0.001;
    
    # 使用する学習モデル
    model = "Model-0-0.pth";
    
    # 学習モデルの保存先
    model_dir = "./";