import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 detector on the Isaac Sim dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "dataset.yaml",
        help="Path to the dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Starting model or checkpoint path.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Training device, such as '0' for GPU or 'cpu'.",
    )
    parser.add_argument(
        "--name",
        default="mini_project_yolov8",
        help="Run name under runs/detect/.",
    )
    args = parser.parse_args()

    data_path = args.data.resolve()
    if not data_path.is_file():
        parser.error(f"Dataset YAML does not exist: {data_path}")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.imgsz < 1:
        parser.error("--imgsz must be at least 1")
    if args.batch < 1:
        parser.error("--batch must be at least 1")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(PROJECT_DIR / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
