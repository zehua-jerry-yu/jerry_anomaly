def train_yolo():
    from ultralytics import YOLO
    # Train the model
    for i in range(1, 6):
        model = YOLO("yolov5n.pt", verbose=True)
        model.train(
            data=f"/root/autodl-tmp/kaggle/anomaly/ssgd/SSGD/annotations_lb101/fold{i}.yaml",
            epochs=100,
            warmup_epochs=10,
            optimizer='AdamW',
            cos_lr=True,
            lr0=5e-4,
            lrf=0.02,
            imgsz=640,
            device="0",
            weight_decay=0.01,
            batch=16,
            scale=0.2,
            flipud=0.5,
            fliplr=0.5,
            degrees=20,
            shear=5,
            mixup=0.01,
            copy_paste=0.2,
            copy_paste_mode="mixup", 
            patience=10, 
            dropout=0.1, 
            seed=0
        )


if __name__ == "__main__":
    train_yolo()