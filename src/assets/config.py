class config:
    
    # 学習させるデータセット
    dataset = "C:\\Enviroments\\nnForge\\src\\dataset\\images\\SDNET2018\\W";
    
    #画像サイズ
    img_size = 224;
    
    # バッチサイズ
    batch_size = 64;
    
    # 学習回数
    epochs = 15;
    
    # 学習率
    learning_rate = 0.0005;
 
    # 使用する学習モデル
    model = f"Model-{epochs}-{batch_size}-{learning_rate}.pth";
    
    # 学習モデルの保存先
    model_dir = "./";
       
    
    # ----- lr_scheduler ----- #
    
    type = "CosineAnnealingLR"
    step_size = 15
    gamma = 0.1
    
    # ------------------------ #
    
    