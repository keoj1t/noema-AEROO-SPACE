from pathlib import Path
import argparse
import sys

from ultralytics import YOLO


def _install_numpy_legacy_aliases() -> None:
    """Compatibility for checkpoints serialized with numpy._core module paths."""
    try:
        import numpy.core as core  # type: ignore
        import numpy.core.multiarray as multiarray  # type: ignore
        import numpy.core.umath as umath  # type: ignore
        import numpy.core.numerictypes as numerictypes  # type: ignore
        import numpy.core._multiarray_umath as _multiarray_umath  # type: ignore

        sys.modules.setdefault("numpy._core", core)
        sys.modules.setdefault("numpy._core.multiarray", multiarray)
        sys.modules.setdefault("numpy._core.umath", umath)
        sys.modules.setdefault("numpy._core.numerictypes", numerictypes)
        sys.modules.setdefault("numpy._core._multiarray_umath", _multiarray_umath)
    except Exception:
        pass


def train_rocket_model(epochs: int, imgsz: int, resume: bool) -> None:
    _install_numpy_legacy_aliases()
    root = Path(__file__).resolve().parent
    base_model = root / "yolov8n.pt"
    last_checkpoint = root / "runs" / "detect" / "train" / "weights" / "last.pt"
    data_yaml = root / "data.yaml"

    if resume and last_checkpoint.exists():
        model = YOLO(str(last_checkpoint))
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=-1,
        )
        return

    model = YOLO(str(base_model))
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=-1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or resume Rocket YOLO model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for fresh run")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from runs/detect/train/weights/last.pt if it exists",
    )
    args = parser.parse_args()

    train_rocket_model(epochs=args.epochs, imgsz=args.imgsz, resume=args.resume)
