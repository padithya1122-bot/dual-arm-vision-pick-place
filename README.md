# dual-arm-vision-pick-place
AI-guided vision system for a dual-arm robotic manipulator using YOLOv8 for object detection, pose estimation, and automated pick-and-place operation.

## YOLOv8 mini-project

A starter computer-vision project for the mini project **AI-Based Vision Guided Pick-and-Place Robot (Dual-Arm Manipulator)**. It trains a YOLOv8 detector on RGB images and YOLO bounding-box labels exported from NVIDIA Isaac Sim, evaluates the trained model, and runs detection on new images.

This project implements the vision and object-detection part of the system. Converting image detections into 3D grasp poses and synchronizing physical robot arms are separate integration steps.

### Requirements

- Python 3.9 or later
- Isaac Sim RGB images and matching YOLO-format label files
- NVIDIA GPU is optional; training can also run on CPU

### Project layout

```text
.
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── runs/                       # Generated training and prediction results
├── dataset.yaml
├── train.py
├── validate.py
├── predict.py
└── requirements.txt
```

Place each image in the appropriate `images` directory and its label file in the matching `labels` directory. For example:

```text
dataset/images/train/cube_001.jpg
dataset/labels/train/cube_001.txt
```

Each label file contains one object per line:

```text
class_id x_center y_center width height
```

The four box values must be normalized to the range 0–1. Empty label files are valid for images with no objects. The class IDs must match the names and order in `dataset.yaml`.

### Install

From the repository root, create and activate a virtual environment (recommended), then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Train

```powershell
python train.py
```

By default, training starts from `yolov8n.pt`, uses `dataset.yaml`, and saves outputs under `runs/detect/mini_project_yolov8/`. The best checkpoint is:

```text
runs/detect/mini_project_yolov8/weights/best.pt
```

Change the training settings from the command line when needed:

```powershell
python train.py --epochs 100 --imgsz 640 --batch 16 --device 0
```

Use `--device cpu` to explicitly train on CPU. See `python train.py --help` for all options.

### Evaluate

Training performs validation during training. To evaluate a saved checkpoint separately:

```powershell
python validate.py --weights runs/detect/mini_project_yolov8/weights/best.pt
```

The script reports the Ultralytics validation metrics, including precision, recall, and mean average precision.

### Predict

Run prediction on one image:

```powershell
python predict.py --source path\to\test_image.jpg
```

Or on a directory of images:

```powershell
python predict.py --source path\to\test_images
```

Predictions are printed with class, confidence, and pixel bounding-box coordinates. Annotated images are saved in `runs/detect/predictions/`. Adjust the confidence threshold with `--conf`, for example:

```powershell
python predict.py --source path\to\test_image.jpg --conf 0.4
```

### Dataset configuration

Edit `dataset.yaml` to match your dataset location and object classes. The included configuration expects two classes (`cube` and `cylinder`); update it if your Isaac Sim dataset uses different classes.
