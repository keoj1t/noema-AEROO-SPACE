from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parent
    data_yaml = root / "data.yaml"
    run_name = "train4_obb_retrain"
    model_name = "yolov8n-obb.pt"

    print(f"Dataset: {data_yaml}")
    print(f"Base model: {model_name}")
    print(f"Run name: {run_name}")

    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=35,
        imgsz=640,
        batch=2,
        device="cpu",
        workers=0,
        patience=20,
        optimizer="AdamW",
        lr0=0.003,
        lrf=0.03,
        cos_lr=True,
        close_mosaic=10,
        mosaic=0.4,
        mixup=0.05,
        copy_paste=0.1,
        degrees=2.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.5,
        project=str(root / "runs" / "detect"),
        name=run_name,
        exist_ok=True,
    )

    print("Training finished.")
    print(f"Best checkpoint: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"Last checkpoint: {results.save_dir / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
