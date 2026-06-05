class config:
    
    # 学習させるデータセット
    dataset = "D:\\Enviroments\\nnForge\\src\\dataset\\images\\SDNET2018\\W";
    
    #画像サイズ
    img_size = 224;
    
    # バッチサイズ
    batch_size = 512;
    
    # 学習回数
    epochs = 50;
    
    # 学習率
    learning_rate = 0.001;
 
    # 使用する学習モデル
    model = f"Model-{epochs}-{batch_size}.pth";
    
    # 学習モデルの保存先
    model_dir = "./";
       
    
    # ----- lr_scheduler ----- #
    
    type = "CosineAnnealingLR"
    step_size = 50
    gamma = 0.1
    
    # ------------------------ #
    
    