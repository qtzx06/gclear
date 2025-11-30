#!/usr/bin/env python3
"""Test OCR on debug frames."""

from PIL import Image, ImageOps
import pytesseract
import re
from pathlib import Path

from data_collection.config import ROIS

def test_ocr(img_path: str):
    img = Image.open(img_path)
    print(f"Image: {img_path} ({img.size})")

    # Timer ROI
    x, y, w, h = ROIS["timer"]
    timer_roi = img.crop((x, y, x + w, y + h))
    timer_roi.save("test_timer_roi.png")

    # Try OCR on timer
    timer_gray = timer_roi.convert("L")
    timer_inv = ImageOps.invert(timer_gray)
    timer_inv.save("test_timer_processed.png")

    timer_text = pytesseract.image_to_string(
        timer_inv,
        config="--psm 7 -c tessedit_char_whitelist=0123456789:"
    ).strip()
    print(f"Timer raw: '{timer_text}'")

    match = re.search(r"(\d{1,2}:\d{2})", timer_text)
    print(f"Timer parsed: {match.group(1) if match else None}")

    # Camp info ROI
    x, y, w, h = ROIS["camp_info"]
    camp_roi = img.crop((x, y, x + w, y + h))
    camp_roi.save("test_camp_roi.png")

    camp_gray = camp_roi.convert("L")
    camp_text = pytesseract.image_to_string(
        camp_gray,
        config="--psm 7 -c tessedit_char_whitelist=0123456789"
    ).strip()
    print(f"Camp raw: '{camp_text}'")

    nums = re.findall(r"\d+", camp_text)
    print(f"Camp health: {int(nums[0]) if nums else None}")


if __name__ == "__main__":
    # Find a debug frame
    frames = list(Path("debug_frames").glob("*.png"))
    if frames:
        test_ocr(str(frames[0]))
    else:
        # Try runs dir
        runs = list(Path("runs").glob("*/frames/*.png"))
        if runs:
            test_ocr(str(runs[0]))
        else:
            print("No debug frames found!")
