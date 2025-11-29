#!/usr/bin/env python3
"""Test local YOLO model on frames."""

from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "models/hecarim.pt"  # or "runs/hecarim_v1/weights/best.pt" if training locally

def test_inference():
    model = YOLO(MODEL_PATH)

    # Test on frames
    test_images = list(Path("data/raw/hecarim_clear").glob("*.png"))[:5]

    if not test_images:
        print("No test images found!")
        return

    for img_path in test_images:
        print(f"\n{img_path.name}:")
        results = model.predict(str(img_path), conf=0.4, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = model.names[int(box.cls)]
                conf = float(box.conf)
                x, y = box.xywh[0][:2].tolist()
                print(f"  {cls}: {conf:.2f} at ({int(x)}, {int(y)})")

if __name__ == "__main__":
    test_inference()
