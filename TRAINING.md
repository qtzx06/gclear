# gclear - Training Guide

Training YOLOv8/v11 for League of Legends jungle clear bot.

## Project Goal

Detect game elements for automated jungle clearing:
- Player position (screen + minimap)
- Jungle camps to click on

## Classes (8)

| Class | What it detects |
|-------|-----------------|
| `player` | Your champion on main screen |
| `mm_player` | Your icon on minimap |
| `blue` | Blue sentinel (blue buff) |
| `red` | Red brambleback (red buff) |
| `gromp` | Gromp |
| `wolves` | Big wolf |
| `raptors` | Big raptor |
| `krugs` | Ancient krug |

## Setup on Linux

```bash
# Clone/copy project
cd gclear

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install deps
pip install ultralytics roboflow

# Set API key
export ROBOFLOW_API_KEY="wPHJmFv21jA1XNv51M4f"
```

## Download Dataset

```python
from roboflow import Roboflow
import os

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("qtzx-olms2").project("hecarim-3ytnh")
version = project.version(1)  # or latest version with train/valid split
dataset = version.download("yolov8", location="datasets/hecarim")
```

Or run:
```bash
python train.py  # will auto-download if datasets/hecarim doesn't exist
```

## Train

```bash
# YOLOv8 nano (fast, good for real-time)
yolo detect train data=datasets/hecarim/data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=16 device=0

# YOLOv8 small (better accuracy)
yolo detect train data=datasets/hecarim/data.yaml model=yolov8s.pt epochs=100 imgsz=640 batch=16 device=0

# YOLOv11 nano (latest)
yolo detect train data=datasets/hecarim/data.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=16 device=0
```

Or use `train.py`:
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="datasets/hecarim/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU index, or "cpu", or "mps" for Mac
    name="hecarim_v1",
    project="runs",
    patience=20,
)
```

## After Training

Best weights saved to: `runs/hecarim_v1/weights/best.pt`

Copy to Mac:
```bash
scp runs/hecarim_v1/weights/best.pt user@mac:~/Desktop/codebase/gclear/models/hecarim.pt
```

## Test Inference

```python
from ultralytics import YOLO

model = YOLO("models/hecarim.pt")
results = model.predict("test_image.png", conf=0.4)

for r in results:
    for box in r.boxes:
        cls = model.names[int(box.cls)]
        conf = float(box.conf)
        x, y, w, h = box.xywh[0].tolist()
        print(f"{cls}: {conf:.2f} at ({x}, {y})")
```

## Notes

- Dataset has 83 base images, augmented to ~247
- All images currently in train split - consider re-generating version with 80/20 train/valid split on Roboflow
- Screen resolution: 1710x1107 (adjust if different on your setup)
- Model trained on 512x512 resized images (Roboflow preprocessing)

## Roboflow Project

- Workspace: `qtzx-olms2`
- Project: `hecarim-3ytnh`
- URL: https://universe.roboflow.com/qtzx-olms2/hecarim-3ytnh
