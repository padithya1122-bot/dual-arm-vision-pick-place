import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (
    PROJECT_DIR / "runs" / "detect" / "mini_project_yolov8" / "weights" / "best.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect objects in an image or directory of images."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Image file or directory of images.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum detection confidence (0–1).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    args = parser.parse_args()

    weights_path = args.weights.resolve()
    source_path = args.source.resolve()
    if not weights_path.is_file():
        parser.error(
            f"Model weights do not exist: {weights_path}. "
            "Train a model first or pass --weights."
        )
    if not source_path.exists():
        parser.error(f"Prediction source does not exist: {source_path}")
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    if args.imgsz < 1:
        parser.error("--imgsz must be at least 1")

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=str(PROJECT_DIR / "runs" / "detect"),
        name="predictions",
        exist_ok=True,
    )

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            print(f"{result.path}: no objects detected")
            continue

        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(
                f"{result.path}: {class_name} ({confidence:.2f}), "
                f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
            )


if __name__ == "__main__":
    main()
