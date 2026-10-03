import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (
    PROJECT_DIR / "runs" / "detect" / "mini_project_yolov8" / "weights" / "best.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLOv8 model on the validation split."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to model weights.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "dataset.yaml",
        help="Path to the dataset YAML file.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument(
        "--device",
        default=None,
        help="Validation device, such as '0' for GPU or 'cpu'.",
    )
    args = parser.parse_args()

    weights_path = args.weights.resolve()
    data_path = args.data.resolve()
    if not weights_path.is_file():
        parser.error(f"Model weights do not exist: {weights_path}")
    if not data_path.is_file():
        parser.error(f"Dataset YAML does not exist: {data_path}")

    model = YOLO(str(weights_path))
    model.val(data=str(data_path), imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
