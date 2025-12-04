# Screen and ROI configuration for League of Legends data collection

SCREEN_WIDTH = 1710
SCREEN_HEIGHT = 1107
GAME_REGION = (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

# ROI definitions (left, top, width, height)
ROIS = {
    "camp_info": (132, 72, 104, 36),
    "timer": (1648, 64, 50, 32),
    "minimap": (1336, 658, 370, 360),
}

# Minimap zones (relative to minimap ROI) - (x1, y1, x2, y2)
MINIMAP_ZONES = {
    "blue_buff": (72, 160, 116, 200),    # shifted left 10
    "gromp": (52, 152, 92, 190),
    "wolves": (86, 194, 120, 230),
    "raptors": (158, 218, 192, 252),
    "red_buff": (166, 258, 204, 290),    # shifted left 10, down 8
    "krugs": (192, 274, 224, 310),
    "spawn": (0, 250, 124, 364),
}

# Colors for ROI visualization (R, G, B, A)
ROI_COLORS = {
    "camp_info": (255, 100, 100, 180),   # Red
    "timer": (100, 255, 100, 180),        # Green
    "minimap": (100, 100, 255, 180),      # Blue
}

# Data collection settings
from pathlib import Path
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
