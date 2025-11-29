# Screen and ROI configuration for League of Legends data collection

SCREEN_WIDTH = 1710
SCREEN_HEIGHT = 1107
GAME_REGION = (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

# ROI definitions (left, top, width, height)
ROIS = {
    # Top-left: Camp health bar + target portrait
    "camp_info": (5, 60, 230, 90),

    # Top-right: Game timer
    "timer": (SCREEN_WIDTH - 100, 60, 90, 30),

    # Bottom-right: Minimap
    "minimap": (SCREEN_WIDTH - 370, SCREEN_HEIGHT - 450, 370, 360),
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
