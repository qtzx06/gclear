#!/usr/bin/env python3
"""Download dataset from Roboflow and train YOLOv8 locally."""

import os
from roboflow import Roboflow
from ultralytics import YOLO
from pathlib import Path

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
WORKSPACE = "qtzx-olms2"
PROJECT = "hecarim-3ytnh"
VERSION = 1

def download_dataset():
    """Download dataset in YOLOv8 format."""
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)

    # Download in YOLOv8 format
    dataset = version.download("yolov8", location="datasets/hecarim")
    return dataset

def train():
    """Train YOLOv8n on the dataset."""
    # Use YOLOv8 nano - fast and good for real-time
    model = YOLO("yolov8n.pt")

    # Train - use MPS (Apple GPU) for speed
    results = model.train(
        data="datasets/hecarim/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="mps",  # Apple GPU
        name="hecarim_v1",
        project="runs",
        patience=15,
        exist_ok=True,
    )

    print(f"\nTraining complete!")
    print(f"Best model saved to: runs/hecarim_v1/weights/best.pt")
    return results

if __name__ == "__main__":
    # Only download if not already present
    if not Path("datasets/hecarim/data.yaml").exists():
        print("Downloading dataset from Roboflow...")
        download_dataset()

    print("Starting training...")
    train()
