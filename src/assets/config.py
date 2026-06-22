class config:
    
    # 学習させるデータセット
    dataset = "C:\\Enviroments\\nnForge\\src\\dataset\\images\\SDNET2018\\W";
    
    #画像サイズ
    img_size = 224;
    
    # バッチサイズ
    batch_size = 128;
    
    # 学習回数
    epochs = 30;
    
    # 学習率
    learning_rate = 0.0001;
 
    # 使用する学習モデル
    model = f"Model-{epochs}-{batch_size}-{learning_rate}.pth";
    
    # 学習モデルの保存先
    model_dir = "./";
       
    
    # ----- lr_scheduler ----- #
    
    type = "CosineAnnealingLR"
    step_size = epochs
    gamma = 0.5
    
    # ------------------------ #
    
    