#!/usr/bin/env python3
"""
State machine for automated jungle clearing.
"""

import time
import subprocess
import re
import os
import math
import random
import threading
from queue import Queue, Empty
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
from strategist import JungleStrategist
from overlay import init_overlay, log_overlay, log_strategist, strategist_thinking, process_events
from som import draw_som_overlay, get_region

DEBUG_VIEW = False  # Show detection window (blocks control)
DEBUG_RECORD = True  # Save debug frames
RUNS_DIR = "runs"

# Strategist config
USE_STRATEGIST = True  # Enabled - Grok verifies camp clears
GROK_QUEUE_SIZE = 2  # Max pending frames for Grok


class State(Enum):
    # Early game states
    STARTUP_LEVELUP = auto()    # Level up Q at game start
    STARTUP_BUY = auto()        # Buy starter item
    WALKING_TO_CAMP = auto()    # Walking via minimap
    WAITING_FOR_SPAWN = auto()  # At camp, waiting for 1:30

    # Combat states
    ENGAGING = auto()           # Moving to attack camp
    ATTACKING = auto()          # In combat, spamming abilities
    KITING = auto()             # Low HP, kiting toward next camp

    # Transition states
    CAMP_CLEARED = auto()       # Camp dead, transitioning
    IDLE = auto()               # Nothing to do


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

# Camps that need spam clicking (multi-monster camps)
SPAM_CLICK_CAMPS = {"wolves", "raptors", "krugs"}

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

# Ability cooldowns
Q_COOLDOWN = 3.9  # Q level 1 (slightly padded)
W_COOLDOWN = 12.73

# Game timing constants
CAMP_SPAWN_TIME = 90  # 1:30 in seconds
EARLY_GAME_END = 15   # First 15 seconds for buying

# Camp max HP values (hardcoded for reliable filtering)
CAMP_MAX_HP = {
    "blue_buff": 2300,
    "gromp": 2050,
    "wolves": 1600,
    "raptors": 1200,
    "red_buff": 2300,
    "krugs": 1400,
}


def parse_game_time(time_str: str) -> int:
    """Parse game time string 'M:SS' to seconds."""
    if not time_str:
        return 0
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except:
        pass
    return 0

# Kiting config
KITE_THRESHOLD = 30.0  # Start kiting below this HP %
ATTACK_MOVE_INTERVAL = 0.3  # Seconds between A-clicks while kiting
HP_SMOOTHING_FRAMES = 5  # Average HP over this many frames before triggering kite


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
        self.last_w_time = 0  # Track W cooldown
        self.in_combat = False
        self.last_attack_move_time = 0  # Track attack-move timing for kiting
        self.camp_selected = False  # Track if we've left-clicked to show health bar
        self.hp_history = []  # Track recent HP readings for smoothing
        self.initial_engage_done = False  # Track if initial attack_click done
        self.is_kiting = False  # Sticky kiting flag - once triggered, stays until camp cleared
        self.no_detection_frames = 0  # Count frames without camp detection (for robust clear detection)
        self.last_known_hp = None  # Last known HP reading for fallback
        self.camp_engage_time = None  # When we started fighting current camp
        self.grok_verified_clear = False  # Grok confirmed camp is dead
        self.combat_frames = 0  # Count frames in combat for Grok timing
        self.last_grok_request_frame = 0  # Track when we last asked Grok

        # Early game tracking
        self.leveled_up_q = False
        self.bought_item = False
        self.pressed_hotkey = False  # Ctrl+Shift+O after buying

        # Strategist (threaded)
        self.strategist = JungleStrategist() if USE_STRATEGIST else None
        self.last_grok_decision = None
        self.last_grok_message = None  # For main thread to display
        self.grok_decision_lock = threading.Lock()
        self.grok_queue = Queue(maxsize=GROK_QUEUE_SIZE)
        self.grok_thread = None
        self.grok_running = False

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

    def get_minimap_zone_center(self, zone_name: str) -> tuple[int, int]:
        """Get minimap-relative center coords for a zone."""
        x1, y1, x2, y2 = MINIMAP_ZONES[zone_name]
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def distance_to_camp(self, mm_pos: tuple[int, int], camp_name: str) -> float:
        """Calculate distance from mm_player to a camp zone on minimap."""
        if not mm_pos or camp_name not in MINIMAP_ZONES:
            return 999
        camp_center = self.get_minimap_zone_center(camp_name)
        dx = mm_pos[0] - camp_center[0]
        dy = mm_pos[1] - camp_center[1]
        return math.sqrt(dx*dx + dy*dy)

    def compute_state(
        self,
        game_time_str: str,
        current_zone: str,
        target_camp: str,
        camp_detection: Optional[Detection],
        camp_health: Optional[dict]
    ) -> State:
        """Determine current state based on game conditions (hardcoded logic)."""
        game_time = parse_game_time(game_time_str)

        # --- EARLY GAME (before camps spawn) ---
        if game_time < EARLY_GAME_END:
            # Level up Q first
            if not self.leveled_up_q:
                return State.STARTUP_LEVELUP
            # Then buy item
            if not self.bought_item:
                return State.STARTUP_BUY

        # --- PRE-SPAWN (walking to camp, waiting) ---
        if game_time < CAMP_SPAWN_TIME:
            # Are we at the target camp zone?
            if current_zone == target_camp:
                return State.WAITING_FOR_SPAWN
            else:
                return State.WALKING_TO_CAMP

        # --- CAMPS HAVE SPAWNED ---

        # If Grok verified camp is dead, transition to cleared
        if self.grok_verified_clear:
            return State.CAMP_CLEARED

        # Sticky kiting - once triggered, stay in kiting until Grok confirms cleared
        if self.is_kiting:
            if camp_detection:
                self.no_detection_frames = 0
                return State.KITING
            else:
                # No detection while kiting - need Grok to verify
                self.no_detection_frames += 1
                # Don't auto-transition, wait for Grok verification
                return State.KITING

        # Is target camp visible on screen?
        if camp_detection:
            self.no_detection_frames = 0  # Reset counter
            # Do we have it targeted (health bar showing)?
            if camp_health:
                # Track when we started fighting
                if self.camp_engage_time is None:
                    self.camp_engage_time = time.time()

                # Use 5-frame smoothed HP for kite decision
                smoothed_hp = self.get_smoothed_hp(camp_health)
                if smoothed_hp is not None and smoothed_hp < KITE_THRESHOLD and self.get_next_camp_zone():
                    self.is_kiting = True  # Set sticky flag
                    return State.KITING
                return State.ATTACKING
            else:
                # Camp visible but not targeted - engage it
                return State.ENGAGING
        else:
            # Camp not visible - count frames
            self.no_detection_frames += 1

            if current_zone == target_camp:
                # We're at the zone but camp not detected - wait for Grok verification
                if self.in_combat or self.is_kiting:
                    # Stay in current state, Grok will verify if dead
                    return State.KITING if self.is_kiting else State.ATTACKING
                else:
                    return State.ENGAGING  # Try to re-engage
            else:
                # Need to walk there
                return State.WALKING_TO_CAMP

        return State.IDLE

    def execute_state(
        self,
        state: State,
        camp_detection: Optional[Detection],
        camp_health: Optional[dict],
        mm_pos: Optional[tuple],
        detections: list
    ):
        """Execute actions based on current state (hardcoded behavior)."""

        if state == State.STARTUP_LEVELUP:
            # Level up Q
            self.level_up_ability("Q")
            self.leveled_up_q = True
            log_overlay("Leveled up Q")

        elif state == State.STARTUP_BUY:
            # Buy starter item
            self._buy_starter_item()
            self.bought_item = True
            log_overlay("Bought starter item")

        elif state == State.WALKING_TO_CAMP:
            # Level up W when walking to gromp (after blue)
            if self.target_camp == "gromp" and self.camp_index == 1:
                self.level_up_ability("W")

            # Click minimap to walk to target camp
            if self.can_act():
                region = get_region(f"mm_{self.target_camp.replace('_buff', '')}")
                if region:
                    self.click(region.x, region.y, right=True)
                    log_overlay(f"Walking to {self.target_camp}")

        elif state == State.WAITING_FOR_SPAWN:
            # Just wait, maybe position near camp
            pass  # Do nothing, camps not spawned yet

        elif state == State.ENGAGING:
            # Camp visible, need to click it to target
            if camp_detection and self.can_act():
                self.click(camp_detection.x, camp_detection.y, right=False)  # Left click to target
                self.camp_selected = True
                log_overlay(f"Targeting {self.target_camp}")

        elif state == State.ATTACKING:
            # In combat - spam attacks and abilities
            # Find player position from detections
            player_pos = None
            for d in detections:
                if d.cls == "player":
                    player_pos = (d.x, d.y)
                    break

            if camp_detection:
                if not self.initial_engage_done:
                    # Initial attack
                    use_spam = self.target_camp in SPAM_CLICK_CAMPS
                    self.attack_click(camp_detection.x, camp_detection.y, spam=use_spam)
                    self.initial_engage_done = True
                    self.in_combat = True
                else:
                    # Continue attacking with A-clicks near player
                    self.attack_move_click(camp_detection.x, camp_detection.y, player_pos=player_pos)

                # Use abilities on cooldown
                self.use_q_if_ready()
                self.use_w_if_ready()

                # Spam smite at blue buff and krugs
                if self.target_camp in ("blue_buff", "krugs"):
                    self.press_key("f")

        elif state == State.KITING:
            # Low HP, kite toward next camp
            next_camp = self.get_next_camp_zone()
            if next_camp and mm_pos:
                kite_x, kite_y = self.get_kite_direction(mm_pos, next_camp)
                self.kite_move(kite_x, kite_y)
                self.use_q_if_ready()

        elif state == State.CAMP_CLEARED:
            # Camp is dead, advance to next
            self._finish_camp()
            log_overlay(f"Cleared! Next: {self.target_camp}")

    def find_camp_on_screen(self, detections: list[Detection], camp_name: str) -> Optional[Detection]:
        """Find a specific camp in detections. Returns the largest one (for multi-monster camps)."""
        target_cls = CAMP_CLASSES.get(camp_name)
        if not target_cls:
            return None

        # Find all matching detections
        matches = [d for d in detections if d.cls == target_cls]
        if not matches:
            return None

        # Return the largest one (big wolf/raptor instead of small ones)
        return max(matches, key=lambda d: d.w * d.h)

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
        """Read camp health from camp_info ROI. Returns dict with current, max, percent.

        NEVER returns None while in combat - uses fallback values instead.
        """
        expected_max = CAMP_MAX_HP.get(self.target_camp, 2300)

        try:
            roi = self.get_roi_image(img, "camp_info")

            # Better preprocessing for health bar
            roi = roi.convert("L")
            roi = ImageOps.autocontrast(roi)
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
                # If no slash found, try to find just a current HP number
                nums = re.findall(r"\d+", text)
                if nums:
                    for num_str in nums:
                        num = int(num_str)
                        if 0 < num <= expected_max:
                            current = num
                            max_hp = expected_max
                            break
                        elif len(num_str) >= 4:
                            mid = len(num_str) // 2
                            first, second = int(num_str[:mid]), int(num_str[mid:])
                            if abs(second - expected_max) < expected_max * 0.2:
                                current = first
                                max_hp = expected_max
                                break

            if current is not None and max_hp is not None and max_hp > 0:
                # Use hardcoded max HP if close to expected
                if abs(max_hp - expected_max) < expected_max * 0.3:
                    max_hp = expected_max

                # Cap current at max
                current = min(current, max_hp)
                current = max(0, current)

                # Current HP should never increase during combat
                if self.in_combat and self.last_known_hp is not None:
                    if current > self.last_known_hp["current"] * 1.1:
                        current = int(self.last_known_hp["current"] * 0.95)

                percent = round((current / max_hp) * 100, 1)
                result = {"current": current, "max": max_hp, "percent": percent, "text": f"{current}/{max_hp}"}
                self.last_known_hp = result
                return result

        except Exception as e:
            print(f"OCR error: {e}")

        # --- FALLBACK LOGIC: Never return None while in combat ---
        if self.in_combat or self.is_kiting:
            if self.last_known_hp is not None:
                # Estimate HP decay (lose ~3% per frame while fighting)
                last_pct = self.last_known_hp["percent"]
                new_pct = max(0, last_pct - 3.0)
                new_current = int((new_pct / 100) * expected_max)
                result = {"current": new_current, "max": expected_max, "percent": new_pct, "text": f"{new_current}/{expected_max}*"}
                self.last_known_hp = result
                return result
            else:
                # No history - assume full HP
                result = {"current": expected_max, "max": expected_max, "percent": 100.0, "text": f"{expected_max}/{expected_max}*"}
                self.last_known_hp = result
                return result

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

    def attack_click(self, x: int, y: int, spam: bool = False):
        """Left-click then right-click for initial engagement (shows health + attacks).
        If spam=True, clicks multiple spots around the target for better hit chance.
        """
        if not self.can_act():
            return

        if spam:
            # Spam clicks in a pattern around the target
            offsets = [(0, 0), (-20, -15), (20, -15), (-20, 15), (20, 15)]
            for ox, oy in offsets:
                cx, cy = x + ox, y + oy
                subprocess.run(["/opt/homebrew/bin/cliclick", f"c:{cx},{cy}", f"rc:{cx},{cy}"], check=True)
                time.sleep(0.05)
            print(f"  -> Spam attack at ({x}, {y})")
        else:
            # Single left+right click
            subprocess.run(["/opt/homebrew/bin/cliclick", f"c:{x},{y}", f"rc:{x},{y}"], check=True)
            print(f"  -> Attack click at ({x}, {y})")

        self.last_action_time = time.time()

    def attack_move_click(self, x: int, y: int, player_pos: tuple = None):
        """A + left-click near player position with small movement."""
        if not self.can_act():
            return

        # Use player position if available, otherwise target position
        px, py = player_pos if player_pos else (x, y)

        # A-clicks close to player with small random offset
        for _ in range(3):
            ox = random.randint(-30, 30)
            oy = random.randint(-30, 30)
            cx, cy = px + ox, py + oy
            subprocess.run(["/opt/homebrew/bin/cliclick", "t:a", f"c:{cx},{cy}"], check=True)
            time.sleep(0.04)

        self.last_action_time = time.time()

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

    def press_ctrl_shift_o(self):
        """Press Ctrl+Shift+O hotkey."""
        try:
            # cliclick: kd/ku for key down/up, ctrl=control, shift=shift
            subprocess.run(["/opt/homebrew/bin/cliclick", "kd:ctrl", "kd:shift", "t:o", "ku:shift", "ku:ctrl"], check=True)
            print("  -> Pressed Ctrl+Shift+O")
        except Exception as e:
            print(f"  -> Hotkey error: {e}")

    def use_q_if_ready(self):
        """Use Q if off cooldown."""
        if time.time() - self.last_q_time >= Q_COOLDOWN:
            self.press_ability("Q")
            self.last_q_time = time.time()

    def use_w_if_ready(self):
        """Use W if off cooldown."""
        if time.time() - self.last_w_time >= W_COOLDOWN:
            self.press_ability("W")
            self.last_w_time = time.time()
            print("  -> Used W")

    def get_smoothed_hp(self, camp_health: Optional[dict]) -> Optional[float]:
        """Get smoothed HP percentage based on recent readings."""
        HP_DECAY_PER_FRAME = 5.0  # Estimated HP loss per frame when OCR fails (faster decay)

        valid_reading = False
        if camp_health and camp_health.get("percent") is not None:
            hp = camp_health["percent"]

            # Check if reading is valid
            if 0 <= hp <= 100:
                if self.hp_history:
                    last_hp = self.hp_history[-1]
                    # HP can only go down (or stay same), and not too fast
                    if hp <= last_hp + 5 and last_hp - hp <= 30:
                        valid_reading = True
                        self.hp_history.append(hp)
                else:
                    # First reading - accept if reasonable
                    valid_reading = True
                    self.hp_history.append(hp)

        # If bad/no reading, estimate by decreasing from last known HP
        if not valid_reading and self.hp_history:
            estimated_hp = max(0, self.hp_history[-1] - HP_DECAY_PER_FRAME)
            self.hp_history.append(estimated_hp)

        # Keep only recent frames
        if len(self.hp_history) > HP_SMOOTHING_FRAMES:
            self.hp_history = self.hp_history[-HP_SMOOTHING_FRAMES:]

        return self._current_smoothed_hp()

    def _current_smoothed_hp(self) -> Optional[float]:
        """Calculate current smoothed HP from history."""
        if len(self.hp_history) >= HP_SMOOTHING_FRAMES:
            return sum(self.hp_history) / len(self.hp_history)
        return None

    def reset_hp_history(self, start_full: bool = False):
        """Clear HP history when switching camps. If start_full, initialize at 100%."""
        if start_full:
            # Start with assumed full HP so we have a baseline
            self.hp_history = [100.0, 100.0]
        else:
            self.hp_history = []

    def kite_move(self, x: int, y: int):
        """Kite: right-click to walk, then A+click to attack while moving."""
        if time.time() - self.last_attack_move_time < ATTACK_MOVE_INTERVAL:
            return
        try:
            # Right-click to start walking
            subprocess.run(["/opt/homebrew/bin/cliclick", f"rc:{x},{y}"], check=True)
            # Small delay to start moving
            time.sleep(0.15)
            # A + left-click to attack-move (will attack nearest while walking)
            subprocess.run(["/opt/homebrew/bin/cliclick", "t:a", f"c:{x},{y}"], check=True)
            self.last_attack_move_time = time.time()
            print(f"  -> Kite to ({x}, {y})")
        except Exception as e:
            print(f"  -> Kite error: {e}")

    def get_next_camp_zone(self) -> Optional[str]:
        """Get the next camp in clear order."""
        next_idx = self.camp_index + 1
        if next_idx < len(CLEAR_ORDER):
            return CLEAR_ORDER[next_idx]
        return None

    def get_kite_direction(self, mm_pos: tuple[int, int], next_camp: str) -> tuple[int, int]:
        """Get screen coords to click for kiting towards next camp.

        Calculates direction from player (mm_pos) to next camp on minimap,
        then returns a point on the game screen in that direction.
        """
        # Get next camp center on minimap (relative coords)
        x1, y1, x2, y2 = MINIMAP_ZONES[next_camp]
        camp_x = (x1 + x2) // 2
        camp_y = (y1 + y2) // 2

        # Direction vector from player to camp (on minimap)
        dx = camp_x - mm_pos[0]
        dy = camp_y - mm_pos[1]

        # Normalize and scale to screen distance (click ~300px away from center)
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0:
            dx = dx / dist
            dy = dy / dist

        # Player is roughly center of screen, click in that direction
        screen_center_x = SCREEN_WIDTH // 2
        screen_center_y = SCREEN_HEIGHT // 2
        kite_distance = 600  # pixels from center to click (aggressive kite)

        click_x = int(screen_center_x + dx * kite_distance)
        click_y = int(screen_center_y + dy * kite_distance)

        # Clamp to screen bounds (allow closer to edges)
        click_x = max(20, min(SCREEN_WIDTH - 20, click_x))
        click_y = max(20, min(SCREEN_HEIGHT - 20, click_y))

        return (click_x, click_y)

    def _apply_grok_decision(self, decision: dict, detections: list, mm_pos):
        """Execute Grok's strategic decision."""
        action = decision.get("action", "ATTACK")
        target = decision.get("target") or self.target_camp

        # Update target if Grok says different or we don't have one
        if target and target in CLEAR_ORDER:
            if self.target_camp is None or target != self.target_camp:
                idx = CLEAR_ORDER.index(target)
                self.camp_index = idx
                self.target_camp = target
                self.camp_selected = False
                self.initial_engage_done = False
                self.reset_hp_history()
                print(f"Target set to: {target}")

        # If still no target, default to first camp
        if not self.target_camp:
            self.target_camp = CLEAR_ORDER[0]
            self.camp_index = 0
            print(f"Defaulting to: {self.target_camp}")

        # Get click position - prefer click_id (SoM) over raw coordinates
        click_id = decision.get("click_id")
        click_pos = decision.get("click")
        ability = decision.get("ability")
        level_up = decision.get("level_up")

        # Resolve click_id to coordinates (string IDs like "A_N", "mm_blue")
        if click_id:
            region = get_region(str(click_id))
            if region:
                click_pos = {"x": region.x, "y": region.y}
                print(f"  -> SoM {click_id} @ ({region.x}, {region.y})")

        # Level up ability if specified (Ctrl+key)
        if level_up and level_up in ["Q", "W", "E", "R"]:
            self.level_up_ability(level_up)
            print(f"  -> LEVEL UP {level_up}")

        # Use ability if specified
        if ability:
            if ability == "F":
                # Smite
                self.press_key("f")
                print("  -> SMITE!")
            elif ability in ["Q", "W", "E"]:
                self.press_ability(ability)

        if action == "SELECT":
            # Left-click to focus/select camp (shows health bar)
            camp = self.find_camp_on_screen(detections, self.target_camp)
            if click_pos:
                click_x, click_y = click_pos.get("x", SCREEN_WIDTH // 2), click_pos.get("y", SCREEN_HEIGHT // 2)
            elif camp:
                click_x, click_y = camp.x, camp.y
            else:
                click_x, click_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

            self.click(click_x, click_y, right=False)  # Left click to select
            self.camp_selected = True
            print(f"  -> SELECT at ({click_x}, {click_y})")

        elif action == "ATTACK":
            # Find and attack target
            camp = self.find_camp_on_screen(detections, self.target_camp)

            # Use Grok's click position if provided, otherwise use camp detection
            if click_pos:
                click_x, click_y = click_pos.get("x", SCREEN_WIDTH // 2), click_pos.get("y", SCREEN_HEIGHT // 2)
            elif camp:
                click_x, click_y = camp.x, camp.y
            else:
                click_x, click_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

            if not self.initial_engage_done:
                use_spam = self.target_camp in SPAM_CLICK_CAMPS
                self.attack_click(click_x, click_y, spam=use_spam)
                self.initial_engage_done = True
                self.in_combat = True
            else:
                self.attack_move_click(click_x, click_y)

            self.state = State.ATTACKING

        elif action == "KITE":
            # Start kiting towards next camp
            next_camp = self.get_next_camp_zone()
            if click_pos:
                # Use Grok's click position for kiting
                self.kite_move(click_pos.get("x", SCREEN_WIDTH // 2), click_pos.get("y", SCREEN_HEIGHT // 2))
            elif next_camp and mm_pos:
                kite_x, kite_y = self.get_kite_direction(mm_pos, next_camp)
                self.kite_move(kite_x, kite_y)
            else:
                # No next camp - just A-click
                self.attack_move_click(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            self.state = State.KITING

        elif action == "WALK":
            # Walk to target camp - use Grok's click or minimap
            if click_pos:
                self.click(click_pos.get("x"), click_pos.get("y"), right=True)
            else:
                x, y = self.get_zone_center(self.target_camp)
                self.click(x, y, right=True)
            self.state = State.WALKING_TO_CAMP

        elif action == "SMITE":
            # Smite the target
            self.press_key("f")
            print("  -> SMITE!")
            # Also attack if camp visible
            camp = self.find_camp_on_screen(detections, self.target_camp)
            if camp:
                self.attack_move_click(camp.x, camp.y)

        elif action == "FINISH":
            # Current camp is dead - move to next
            self._finish_camp()

        elif action == "SHOP":
            # Open shop, buy Gustwalker Hatchling, close shop
            self._buy_starter_item()

        elif action == "WAIT":
            # Do nothing - Grok decided to wait
            pass

    def _buy_starter_item(self):
        """Open shop, buy Gustwalker Hatchling, close shop."""
        try:
            print("  -> Opening shop...")
            # Press B to open shop
            subprocess.run(["/opt/homebrew/bin/cliclick", "t:b"], check=True)
            time.sleep(0.8)  # Wait for shop to fully open

            # Gustwalker Hatchling in recommended items
            # Right-click to buy (try multiple times)
            item_pos = (650, 340)  # Between left and center
            print(f"  -> Clicking item at {item_pos}...")
            for i in range(3):
                subprocess.run(["/opt/homebrew/bin/cliclick", f"rc:{item_pos[0]},{item_pos[1]}"], check=True)
                time.sleep(0.2)

            time.sleep(0.3)

            # Close shop with B again
            print("  -> Closing shop...")
            subprocess.run(["/opt/homebrew/bin/cliclick", "t:b"], check=True)
            time.sleep(0.2)

            print("  -> Bought starter item")
        except Exception as e:
            print(f"  -> Shop error: {e}")
            # Try to close shop
            try:
                subprocess.run(["/opt/homebrew/bin/cliclick", "t:b"], check=True)
            except:
                pass

    def _finish_camp(self):
        """Called when a camp is cleared - level up and move to next."""
        # Level up ability after this camp
        ability_to_level = LEVEL_UP_ORDER.get(self.camp_index)
        if ability_to_level:
            self.level_up_ability(ability_to_level)

        # Reset all combat/camp state
        self.in_combat = False
        self.camp_selected = False
        self.initial_engage_done = False
        self.is_kiting = False
        self.no_detection_frames = 0
        self.last_known_hp = None
        self.camp_engage_time = None
        self.grok_verified_clear = False
        self.combat_frames = 0
        self.last_grok_request_frame = 0
        self.reset_hp_history()

        self.camp_index += 1
        if self.camp_index < len(CLEAR_ORDER):
            self.target_camp = CLEAR_ORDER[self.camp_index]
            self.state = State.WALKING_TO_CAMP
            print(f"Next camp: {self.target_camp}")
        else:
            self.state = State.IDLE
            print("Clear complete!")

    def _grok_worker(self):
        """Background thread for Grok camp verification."""
        print("[GROK THREAD] Started - Camp Clear Verifier")
        while self.grok_running:
            try:
                frame_data = self.grok_queue.get(timeout=0.5)
            except Empty:
                continue

            if frame_data is None:  # Shutdown signal
                break

            try:
                # Call Grok to verify if camp is cleared
                result = self.strategist.verify_camp_cleared(
                    img=frame_data["img"],
                    target_camp=frame_data["target_camp"],
                    time_at_camp=frame_data["time_at_camp"],
                    last_hp_percent=frame_data["last_hp_percent"],
                    detections=frame_data["detections"],
                    current_zone=frame_data["current_zone"],
                    game_time=frame_data["game_time"]
                )

                camp_dead = result.get("camp_dead", False)
                confidence = result.get("confidence", 0.0)
                reason = result.get("reason", "")

                # Display result
                status = "DEAD" if camp_dead else "ALIVE"
                grok_msg = f"[GROK] {frame_data['target_camp']}: {status} ({confidence:.0%}) - {reason}"
                print(grok_msg)

                with self.grok_decision_lock:
                    self.last_grok_message = grok_msg
                    # If Grok says camp is dead with high confidence, set flag
                    if camp_dead and confidence >= 0.7:
                        self.grok_verified_clear = True
                        print(f"[GROK] VERIFIED CLEAR: {frame_data['target_camp']}")

            except Exception as e:
                print(f"[GROK THREAD] Error: {e}")
                import traceback
                traceback.print_exc()

        print("[GROK THREAD] Stopped")

    def _start_grok_thread(self):
        """Start the Grok worker thread."""
        if self.strategist and not self.grok_running:
            self.grok_running = True
            self.grok_thread = threading.Thread(target=self._grok_worker, daemon=True)
            self.grok_thread.start()

    def _stop_grok_thread(self):
        """Stop the Grok worker thread."""
        if self.grok_running:
            self.grok_running = False
            # Send shutdown signal
            try:
                self.grok_queue.put(None, timeout=0.1)
            except:
                pass
            if self.grok_thread:
                self.grok_thread.join(timeout=2.0)

    def show_debug(self, img: Image.Image, detections: list[Detection], mm_pos, current_zone, game_time=None, camp_health=None):
        """Show debug window with detections and SoM overlay."""
        # Add SoM overlay to the image (includes detection coords)
        img_with_som = draw_som_overlay(img, detections)

        # Convert to cv2 format
        frame = cv2.cvtColor(np.array(img_with_som), cv2.COLOR_RGB2BGR)

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

        # Show live (optional)
        if DEBUG_VIEW:
            scale = 0.5
            small = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
            cv2.imshow("GClear Debug", small)
            cv2.waitKey(1)

    def update(self):
        """Main update loop - state machine driven, Grok for advisory."""
        img = self.capture_screen()
        detections = self.detect(img)

        # --- FAST DATA GATHERING (every frame) ---
        mm_pos = self.get_minimap_player_pos(detections)
        current_zone = self.get_player_zone(mm_pos) if mm_pos else None
        game_time = self.ocr_timer(img)
        camp_health = self.ocr_camp_health(img)

        # Initialize target if not set
        if not self.target_camp:
            self.target_camp = CLEAR_ORDER[0]

        # Find target camp on screen
        target_camp_detection = self.find_camp_on_screen(detections, self.target_camp)

        # --- COMPUTE STATE (hardcoded logic) ---
        new_state = self.compute_state(
            game_time_str=game_time,
            current_zone=current_zone,
            target_camp=self.target_camp,
            camp_detection=target_camp_detection,
            camp_health=camp_health
        )

        # Log state changes
        if new_state != self.state:
            self.state = new_state
            log_overlay(f"STATE: {self.state.name}")

        # --- EXECUTE STATE ACTIONS ---
        self.execute_state(
            state=self.state,
            camp_detection=target_camp_detection,
            camp_health=camp_health,
            mm_pos=mm_pos,
            detections=detections
        )

        # --- USE W WHEN APPROACHING CAMP (distance-based) ---
        W_APPROACH_DISTANCE = 12  # Minimap pixels - activate W when very close
        if self.camp_index >= 1 and mm_pos:  # After blue, we have W leveled
            dist = self.distance_to_camp(mm_pos, self.target_camp)
            if dist < W_APPROACH_DISTANCE:
                self.use_w_if_ready()

        # --- GROK CAMP VERIFICATION ---
        # Count combat frames and ask Grok periodically
        if self.in_combat or self.is_kiting:
            self.combat_frames += 1

            # Ask Grok every 15 frames during combat, or immediately if no detection
            GROK_INTERVAL = 15  # Ask every 15 frames (~3 seconds at 0.2s tick)
            frames_since_last = self.frame_count - self.last_grok_request_frame
            should_ask = (
                self.strategist and
                not self.grok_verified_clear and
                (frames_since_last >= GROK_INTERVAL or self.no_detection_frames >= 3)
            )

            if should_ask:
                self.last_grok_request_frame = self.frame_count

                # Calculate time fighting this camp
                time_at_camp = 0.0
                if self.camp_engage_time:
                    time_at_camp = time.time() - self.camp_engage_time

                # Get last known HP percent
                last_hp_pct = None
                if self.last_known_hp:
                    last_hp_pct = self.last_known_hp.get("percent")

                frame_data = {
                    "img": img,
                    "target_camp": self.target_camp,
                    "time_at_camp": time_at_camp,
                    "last_hp_percent": last_hp_pct,
                    "detections": detections,
                    "current_zone": current_zone,
                    "game_time": game_time,
                }

                # Clear queue and add new request
                while not self.grok_queue.empty():
                    try:
                        self.grok_queue.get_nowait()
                    except Empty:
                        break
                try:
                    self.grok_queue.put_nowait(frame_data)
                    print(f"  -> Queued Grok verify (combat frame {self.combat_frames})")
                except:
                    pass
        else:
            self.combat_frames = 0  # Reset when not in combat

        # --- DEBUG/LOGGING ---
        if DEBUG_VIEW or DEBUG_RECORD:
            self.show_debug(img, detections, mm_pos, current_zone, game_time, camp_health)

        # Get Grok message for display (advisory only)
        with self.grok_decision_lock:
            grok_message = self.last_grok_message
            self.last_grok_message = None

        if grok_message:
            for line in grok_message.split('\n'):
                log_strategist(line)

        # --- LOGGING ---
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

        # Detection line - show what YOLO sees
        if detections:
            det_strs = [f"{d.cls}@({d.x},{d.y})" for d in detections[:5]]
            det_line = "DETECT: " + " | ".join(det_strs)
        else:
            det_line = "DETECT: (none)"
        log_overlay(det_line)

        # Status line: [time] STATE | target | HP
        hp_str = f"{camp_health['current']}/{camp_health['max']}" if camp_health else "-"
        time_str = game_time or "0:00"
        zone_str = current_zone or "?"

        status_msg = f"[{time_str}] {self.state.name} | {self.target_camp} @ {zone_str} | HP:{hp_str}"
        print(f"{det_line}\n{status_msg}")
        log_overlay(status_msg)

        self.frame_count += 1

    def run(self, tick_rate: float = 0.2):
        """Run the bot loop."""
        print("Starting jungle bot... (Ctrl+C to stop)")
        print(f"Clear order: {CLEAR_ORDER}")
        print(f"Run ID: {self.run_id}")
        print(f"Tick rate: {tick_rate}s ({1/tick_rate:.1f} FPS target)")
        if DEBUG_RECORD:
            print(f"Saving to: {self.run_dir}/")

        # Initialize overlay
        self.overlay = init_overlay()
        log_overlay("GClear Bot Started")
        log_overlay(f"Clear: {' -> '.join(CLEAR_ORDER)}")

        # Start Grok thread
        self._start_grok_thread()
        log_overlay("Grok thread started")

        try:
            while True:
                self.update()
                process_events()  # Keep overlay responsive
                time.sleep(tick_rate)
        except KeyboardInterrupt:
            print("\nStopped.")
            log_overlay("Bot Stopped")
        finally:
            self._stop_grok_thread()
            if self.strategist:
                self.strategist.close()
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
