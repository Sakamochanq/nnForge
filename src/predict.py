from ultralytics import YOLO


if __name__ == '__main__':
    
    while True:
    
        # 任意のものを使用する
        trained_model = input("\nTrained Model ❯  ")
        img = input("\nImage ❯ ")

        model = YOLO(trained_model)

        results = model.predict(
            source=img,
            save=True,
            imgsz=640
        )

        print("Done!")