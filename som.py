#!/usr/bin/env python3
"""
Set-of-Mark (SoM) - Simple version.
- Minimap positions for walking to camps
- YOLO detections provide attack coordinates
"""

from dataclasses import dataclass
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

@dataclass
class Region:
    """A clickable region on screen."""
    id: str
    name: str
    x: int
    y: int
    description: str


# =============================================================================
# MINIMAP CAMP POSITIONS (for WALK action)
# =============================================================================

REGIONS = [
    Region("mm_blue", "Blue Buff", 1440, 838, "Walk to Blue Sentinel"),
    Region("mm_gromp", "Gromp", 1408, 829, "Walk to Gromp"),
    Region("mm_wolves", "Wolves", 1435, 875, "Walk to Wolves"),  # shifted left 4, down 5
    Region("mm_raptors", "Raptors", 1511, 905, "Walk to Raptors"),  # shifted down 12
    Region("mm_red", "Red Buff", 1531, 938, "Walk to Red Brambleback"),  # shifted down 14
    Region("mm_krugs", "Krugs", 1544, 965, "Walk to Krugs"),  # shifted down 15
]

REGION_BY_ID = {r.id: r for r in REGIONS}


def get_region(region_id: str) -> Optional[Region]:
    """Get region by ID."""
    return REGION_BY_ID.get(region_id)


def draw_som_overlay(img: Image.Image, detections: list = None) -> Image.Image:
    """
    Draw SoM markers on image.
    - Minimap positions (green circles)
    - YOLO detections with coordinates (red boxes with coords)
    """
    img = img.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except:
        font = ImageFont.load_default()

    # Draw minimap positions
    for region in REGIONS:
        r = 10
        color = (100, 255, 100)  # Green
        draw.ellipse([region.x - r, region.y - r, region.x + r, region.y + r],
                    outline=color, width=2)
        short_name = region.id.replace("mm_", "")
        draw.text((region.x - 15, region.y + 12), short_name, fill=color, font=font)

    # Draw YOLO detections with their coordinates (so Grok knows where to click)
    if detections:
        for d in detections:
            color = (255, 100, 100)  # Red
            # Draw box
            x1, y1 = d.x - d.w//2, d.y - d.h//2
            x2, y2 = d.x + d.w//2, d.y + d.h//2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            # Show class and CENTER coordinates
            label = f"{d.cls} @({d.x},{d.y})"
            draw.text((x1, y1 - 15), label, fill=color, font=font)

    return img


def get_som_prompt_section() -> str:
    """Generate the SoM section for Grok's prompt."""
    return """=== CLICK SYSTEM ===
For WALK action, use click_id:
  "mm_blue", "mm_gromp", "mm_wolves", "mm_raptors", "mm_red", "mm_krugs"

For ATTACK/SELECT action, use the coordinates from YOLO DETECTIONS.
The detections show: class @(x,y) - use those x,y coords in "click": {"x": ..., "y": ...}
"""


if __name__ == "__main__":
    print("=== Minimap Regions ===")
    for r in REGIONS:
        print(f"  {r.id:12s} @ ({r.x}, {r.y}) - {r.description}")
