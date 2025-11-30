#!/usr/bin/env python3
"""
State machine for automated jungle clearing.
"""

import time
import subprocess
import re
import os
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from ultralytics import YOLO
import Quartz
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import numpy as np
import pytesseract

# Set tesseract path for macOS
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

from data_collection.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, ROIS, MINIMAP_ZONES
)

DEBUG_VIEW = False  # Show detection window (blocks control)
DEBUG_RECORD = True  # Save debug frames
RUNS_DIR = "runs"


class State(Enum):
    IDLE = auto()
    WALK_TO_CAMP = auto()
    APPROACH_CAMP = auto()  # Right-click to get in range
    ATTACK_CAMP = auto()
    KITING = auto()         # Kite while clearing (< 50% HP)
    WAIT_RESPAWN = auto()


@dataclass
class Detection:
    cls: str
    conf: float
    x: int
    y: int
    w: int
    h: int


# Jungle clear order (blue side)
CLEAR_ORDER = ["blue_buff", "gromp", "wolves", "raptors", "red_buff", "krugs"]

# Map camp names to detection classes
CAMP_CLASSES = {
    "blue_buff": "blue",
    "gromp": "gromp",
    "wolves": "wolves",
    "raptors": "raptors",
    "red_buff": "red",
    "krugs": "krugs",
}

# Hecarim ability config
ABILITY_KEYS = {"Q": "q", "W": "w", "E": "e", "R": "r"}
LEVEL_UP_KEYS = {"Q": "ctrl+q", "W": "ctrl+w", "E": "ctrl+e", "R": "ctrl+r"}

# Level up after each camp: camp_index -> ability to level
LEVEL_UP_ORDER = {
    0: None,      # Start with Q already (level 1)
    1: "W",       # After blue -> level W
    2: "Q",       # After gromp -> level Q
    3: "Q",       # After wolves -> level Q
    4: "W",       # After raptors -> level W
    5: "E",       # After red -> level E (level 5)
    6: "Q",       # After krugs -> level Q (level 6)
}

# Q cooldown (Hecarim Q is 4 seconds base)
Q_COOLDOWN = 4.0

# Kiting config
KITE_THRESHOLD = 50.0  # Start kiting below this HP %
ATTACK_MOVE_INTERVAL = 0.3  # Seconds between A-clicks while kiting


class JungleBot:
    def __init__(self, model_path: str = "models/hecarim.pt"):
        self.model = YOLO(model_path)
        self.state = State.IDLE
        self.target_camp: Optional[str] = None
        self.camp_index = 0
        self.last_action_time = 0
        self.action_cooldown = 0.5  # seconds between actions
        self.frame_count = 0
        self.logs = []
        self.last_q_time = 0  # Track Q cooldown
        self.in_combat = False
        self.last_attack_move_time = 0  # Track attack-move timing for kiting
        self.camp_selected = False  # Track if we've left-clicked to show health bar

        # Set up run directory with timestamp
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(RUNS_DIR, self.run_id)
        self.frames_dir = os.path.join(self.run_dir, "frames")

        if DEBUG_RECORD:
            os.makedirs(self.frames_dir, exist_ok=True)

    def capture_screen(self) -> Image.Image:
        """Capture the game screen (scaled to logical resolution)."""
        region = Quartz.CGRectMake(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        image_ref = Quartz.CGWindowListCreateImage(
            region,
            Quartz.kCGWindowListOptionOnScreenOnly,
            Quartz.kCGNullWindowID,
            Quartz.kCGWindowImageDefault
        )

        width = Quartz.CGImageGetWidth(image_ref)
        height = Quartz.CGImageGetHeight(image_ref)
        bytes_per_row = Quartz.CGImageGetBytesPerRow(image_ref)

        data_provider = Quartz.CGImageGetDataProvider(image_ref)
        data = Quartz.CGDataProviderCopyData(data_provider)

        img = Image.frombytes("RGBA", (width, height), data, "raw", "BGRA", bytes_per_row)
        img = img.convert("RGB")

        # Scale down from Retina (2x) to logical resolution
        if width != SCREEN_WIDTH or height != SCREEN_HEIGHT:
            img = img.resize((SCREEN_WIDTH, SCREEN_HEIGHT), Image.Resampling.LANCZOS)

        return img

    def detect(self, img: Image.Image) -> list[Detection]:
        """Run YOLO detection on image."""
        results = self.model.predict(img, conf=0.4, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = self.model.names[int(box.cls)]
                conf = float(box.conf)
                x, y, w, h = box.xywh[0].tolist()
                detections.append(Detection(cls, conf, int(x), int(y), int(w), int(h)))

        return detections

    def get_minimap_player_pos(self, detections: list[Detection]) -> Optional[tuple[int, int]]:
        """Get mm_player position relative to minimap ROI."""
        mm_x, mm_y, mm_w, mm_h = ROIS["minimap"]

        for d in detections:
            if d.cls == "mm_player":
                # Convert to minimap-relative coords
                rel_x = d.x - mm_x
                rel_y = d.y - mm_y
                if 0 <= rel_x <= mm_w and 0 <= rel_y <= mm_h:
                    return (rel_x, rel_y)
        return None

    def get_player_zone(self, mm_pos: tuple[int, int]) -> Optional[str]:
        """Determine which zone the player is in."""
        x, y = mm_pos
        for zone_name, (x1, y1, x2, y2) in MINIMAP_ZONES.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_name
        return None

    def get_zone_center(self, zone_name: str) -> tuple[int, int]:
        """Get screen coords for center of a minimap zone."""
        mm_x, mm_y, _, _ = ROIS["minimap"]
        x1, y1, x2, y2 = MINIMAP_ZONES[zone_name]
        center_x = mm_x + (x1 + x2) // 2
        center_y = mm_y + (y1 + y2) // 2
        return (center_x, center_y)

    def find_camp_on_screen(self, detections: list[Detection], camp_name: str) -> Optional[Detection]:
        """Find a specific camp in detections."""
        target_cls = CAMP_CLASSES.get(camp_name)
        if not target_cls:
            return None

        for d in detections:
            if d.cls == target_cls:
                return d
        return None

    def can_act(self) -> bool:
        """Check if enough time has passed since last action."""
        return time.time() - self.last_action_time >= self.action_cooldown

    def get_roi_image(self, img: Image.Image, roi_name: str) -> Image.Image:
        """Extract ROI from image."""
        x, y, w, h = ROIS[roi_name]
        return img.crop((x, y, x + w, y + h))

    def ocr_timer(self, img: Image.Image) -> Optional[str]:
        """Read game timer from timer ROI."""
        try:
            roi = self.get_roi_image(img, "timer")
            # Save for debugging first few frames
            if self.frame_count < 3:
                roi.save(f"{self.run_dir}/debug_timer_{self.frame_count}.png")
            roi = roi.convert("L")
            roi = ImageOps.invert(roi)
            text = pytesseract.image_to_string(roi, config="--psm 7 -c tessedit_char_whitelist=0123456789:").strip()
            match = re.search(r"(\d{1,2}:\d{2})", text)
            return match.group(1) if match else text if text else None
        except Exception as e:
            print(f"OCR error: {e}")
            return None

    def ocr_camp_health(self, img: Image.Image) -> Optional[dict]:
        """Read camp health from camp_info ROI. Returns dict with current, max, percent."""
        try:
            roi = self.get_roi_image(img, "camp_info")

            # Better preprocessing for health bar
            roi = roi.convert("L")
            # Increase contrast
            roi = ImageOps.autocontrast(roi)
            # Threshold to make text cleaner
            roi = roi.point(lambda x: 255 if x > 128 else 0)

            # OCR with slash allowed
            text = pytesseract.image_to_string(
                roi,
                config="--psm 7 -c tessedit_char_whitelist=0123456789/"
            ).strip()

            current, max_hp = None, None

            # Try to find xx/xx pattern
            match = re.search(r"(\d+)\s*/\s*(\d+)", text)
            if match:
                current, max_hp = int(match.group(1)), int(match.group(2))
            else:
                # If no slash found, try to split a long number in half
                nums = re.findall(r"\d+", text)
                if nums:
                    num_str = nums[0]
                    if len(num_str) >= 4 and len(num_str) % 2 == 0:
                        mid = len(num_str) // 2
                        first, second = num_str[:mid], num_str[mid:]
                        if first == second or abs(int(first) - int(second)) < int(second) * 0.5:
                            current, max_hp = int(first), int(second)

            if current is not None and max_hp is not None and max_hp > 0:
                percent = round((current / max_hp) * 100, 1)
                return {"current": current, "max": max_hp, "percent": percent, "text": f"{current}/{max_hp}"}

            return None
        except Exception:
            return None

    def click(self, x: int, y: int, right: bool = False):
        """Click at screen position using cliclick."""
        if not self.can_act():
            return

        # cliclick: rc = right click, c = left click
        cmd = "rc" if right else "c"
        subprocess.run(["/opt/homebrew/bin/cliclick", f"{cmd}:{x},{y}"], check=True)

        self.last_action_time = time.time()
        btn_name = 'right' if right else 'left'
        print(f"  -> Click ({btn_name}) at ({x}, {y})")

    def press_key(self, key: str):
        """Press a key using cliclick."""
        try:
            # cliclick t: for typing single characters
            subprocess.run(["/opt/homebrew/bin/cliclick", f"t:{key}"], check=True)
            print(f"  -> Key: {key}")
        except Exception as e:
            print(f"  -> Key error: {e}")

    def press_ability(self, ability: str):
        """Press an ability key (Q, W, E, R)."""
        key = ABILITY_KEYS.get(ability)
        if key:
            self.press_key(key)

    def level_up_ability(self, ability: str):
        """Level up an ability (cmd+Q, etc)."""
        try:
            key = ABILITY_KEYS.get(ability)
            if key:
                # cliclick: kd for key down, t for type, ku for key up (cmd = command)
                subprocess.run(["/opt/homebrew/bin/cliclick", "kd:cmd", f"t:{key}", "ku:cmd"], check=True)
                print(f"  -> Level up: {ability}")
        except Exception as e:
            print(f"  -> Level up error: {e}")

    def use_q_if_ready(self):
        """Use Q if off cooldown."""
        if time.time() - self.last_q_time >= Q_COOLDOWN:
            self.press_ability("Q")
            self.last_q_time = time.time()

    def attack_move(self, x: int, y: int):
        """Attack-move: press A, then left-click at position."""
        if time.time() - self.last_attack_move_time < ATTACK_MOVE_INTERVAL:
            return
        try:
            # Press A, then click at position
            subprocess.run(["/opt/homebrew/bin/cliclick", "t:a", f"c:{x},{y}"], check=True)
            self.last_attack_move_time = time.time()
            print(f"  -> Attack-move to ({x}, {y})")
        except Exception as e:
            print(f"  -> Attack-move error: {e}")

    def get_next_camp_zone(self) -> Optional[str]:
        """Get the next camp in clear order."""
        next_idx = self.camp_index + 1
        if next_idx < len(CLEAR_ORDER):
            return CLEAR_ORDER[next_idx]
        return None

    def show_debug(self, img: Image.Image, detections: list[Detection], mm_pos, current_zone, game_time=None, camp_health=None):
        """Show debug window with detections."""
        # Convert to cv2 format
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Draw ROIs
        for name, (x, y, w, h) in ROIS.items():
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw minimap zones
        mm_x, mm_y, _, _ = ROIS["minimap"]
        for zone_name, (x1, y1, x2, y2) in MINIMAP_ZONES.items():
            color = (0, 255, 0) if zone_name == current_zone else (100, 100, 100)
            cv2.rectangle(frame, (mm_x+x1, mm_y+y1), (mm_x+x2, mm_y+y2), color, 1)
            cv2.putText(frame, zone_name[:4], (mm_x+x1, mm_y+y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Draw detections
        for d in detections:
            color = (0, 255, 0) if d.cls == "mm_player" else (0, 0, 255)
            x1, y1 = d.x - d.w//2, d.y - d.h//2
            x2, y2 = d.x + d.w//2, d.y + d.h//2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{d.cls} {d.conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw mm_player position
        if mm_pos:
            abs_x = mm_x + mm_pos[0]
            abs_y = mm_y + mm_pos[1]
            cv2.circle(frame, (abs_x, abs_y), 10, (0, 255, 255), 2)

        # Status text
        cv2.putText(frame, f"State: {self.state.name} | Zone: {current_zone} | Target: {self.target_camp}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        hp_display = f"{camp_health['text']} ({camp_health['percent']}%)" if camp_health else "None"
        cv2.putText(frame, f"Time: {game_time} | HP: {hp_display}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save frame as PNG
        if DEBUG_RECORD:
            cv2.imwrite(f"{self.frames_dir}/frame_{self.frame_count:04d}.png", frame)
            self.frame_count += 1

        # Show live (optional)
        if DEBUG_VIEW:
            scale = 0.5
            small = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
            cv2.imshow("GClear Debug", small)
            cv2.waitKey(1)

    def update(self):
        """Main update loop - run one tick of the state machine."""
        img = self.capture_screen()
        detections = self.detect(img)

        mm_pos = self.get_minimap_player_pos(detections)
        current_zone = self.get_player_zone(mm_pos) if mm_pos else None

        # OCR data
        game_time = self.ocr_timer(img)
        camp_health = self.ocr_camp_health(img)

        if DEBUG_VIEW or DEBUG_RECORD:
            self.show_debug(img, detections, mm_pos, current_zone, game_time, camp_health)

        # Log entry
        log_entry = {
            "frame": self.frame_count,
            "timestamp": time.time(),
            "state": self.state.name,
            "zone": current_zone,
            "target": self.target_camp,
            "game_time": game_time,
            "camp_health": camp_health,
            "detections": [{"cls": d.cls, "conf": d.conf, "x": d.x, "y": d.y} for d in detections],
            "mm_pos": mm_pos,
        }
        self.logs.append(log_entry)

        hp_str = f"{camp_health['text']} ({camp_health['percent']}%)" if camp_health else None
        print(f"State: {self.state.name} | Zone: {current_zone} | Target: {self.target_camp} | Time: {game_time} | HP: {hp_str} | Detections: {[d.cls for d in detections]}")

        if self.state == State.IDLE:
            # Start clearing
            if self.camp_index < len(CLEAR_ORDER):
                self.target_camp = CLEAR_ORDER[self.camp_index]
                self.state = State.WALK_TO_CAMP
                print(f"Starting clear: {self.target_camp}")

        elif self.state == State.WALK_TO_CAMP:
            if current_zone == self.target_camp:
                # Arrived at camp
                self.state = State.ATTACK_CAMP
                print(f"Arrived at {self.target_camp}")
            else:
                # Click minimap to walk
                x, y = self.get_zone_center(self.target_camp)
                self.click(x, y, right=True)

        elif self.state == State.ATTACK_CAMP:
            camp = self.find_camp_on_screen(detections, self.target_camp)
            if camp:
                # Left click first to show health bar, then right click to attack
                if not self.in_combat:
                    self.click(camp.x, camp.y, right=False)  # Left click to select
                    self.in_combat = True
                else:
                    self.click(camp.x, camp.y, right=True)  # Right click to attack
                # Spam Q during combat
                self.use_q_if_ready()
            else:
                # Camp dead or not visible - move to next
                if self.in_combat:
                    # Just finished a camp - level up ability
                    ability_to_level = LEVEL_UP_ORDER.get(self.camp_index)
                    if ability_to_level:
                        self.level_up_ability(ability_to_level)
                    self.in_combat = False

                self.camp_index += 1
                if self.camp_index < len(CLEAR_ORDER):
                    self.target_camp = CLEAR_ORDER[self.camp_index]
                    self.state = State.WALK_TO_CAMP
                    print(f"Next camp: {self.target_camp}")
                else:
                    self.state = State.IDLE
                    print("Clear complete!")

    def run(self, tick_rate: float = 0.5):
        """Run the bot loop."""
        print("Starting jungle bot... (Ctrl+C to stop)")
        print(f"Clear order: {CLEAR_ORDER}")
        print(f"Run ID: {self.run_id}")
        if DEBUG_RECORD:
            print(f"Saving to: {self.run_dir}/")

        try:
            while True:
                self.update()
                time.sleep(tick_rate)
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.save_logs()

    def save_logs(self):
        """Save logs to JSON file."""
        import json
        os.makedirs(self.run_dir, exist_ok=True)
        log_path = os.path.join(self.run_dir, "log.json")
        with open(log_path, "w") as f:
            json.dump({
                "run_id": self.run_id,
                "frames": self.frame_count,
                "logs": self.logs,
            }, f, indent=2)
        print(f"Saved {self.frame_count} frames and {len(self.logs)} log entries to {self.run_dir}/")


def main():
    bot = JungleBot()
    bot.run()


if __name__ == "__main__":
    main()
