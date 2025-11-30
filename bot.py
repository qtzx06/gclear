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
USE_STRATEGIST = False  # Disabled - focus on state machine first
GROK_QUEUE_SIZE = 2  # Max pending frames for Grok (drop old if full)
PLANNER_INTERVAL = 50  # Call deep reasoning planner every N frames


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
Q_COOLDOWN = 3.64  # Q level 1
W_COOLDOWN = 12.73

# Game timing constants
CAMP_SPAWN_TIME = 90  # 1:30 in seconds
EARLY_GAME_END = 15   # First 15 seconds for buying


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
KITE_THRESHOLD = 50.0  # Start kiting below this HP %
ATTACK_MOVE_INTERVAL = 0.3  # Seconds between A-clicks while kiting
HP_SMOOTHING_FRAMES = 2  # Average HP over this many frames before triggering kite


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
        self.expected_max_hp = None  # Track expected max HP for current camp
        self.initial_engage_done = False  # Track if initial attack_click done

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
        self.planner_counter = 0

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
        # Is target camp visible on screen?
        if camp_detection:
            # Do we have it targeted (health bar showing)?
            if camp_health:
                hp_pct = camp_health.get('percent', 100)
                if hp_pct < KITE_THRESHOLD and self.get_next_camp_zone():
                    return State.KITING
                return State.ATTACKING
            else:
                # Camp visible but not targeted - engage it
                return State.ENGAGING
        else:
            # Camp not visible
            if current_zone == target_camp:
                # We're at the zone but camp not detected - it's dead
                return State.CAMP_CLEARED
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
            if camp_detection:
                if not self.initial_engage_done:
                    # Initial attack
                    use_spam = self.target_camp in SPAM_CLICK_CAMPS
                    self.attack_click(camp_detection.x, camp_detection.y, spam=use_spam)
                    self.initial_engage_done = True
                    self.in_combat = True
                else:
                    # Continue attacking with A-clicks
                    self.attack_move_click(camp_detection.x, camp_detection.y)

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

            # Spam level up W while kiting blue buff (blue about to die)
            if self.target_camp == "blue_buff":
                self.level_up_ability("W")
                log_overlay("Level up W (kiting blue)")

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
                # Sanity checks
                if current > max_hp:
                    return None
                if max_hp < 200 or max_hp > 10000:
                    return None

                # Max HP should never increase (use lowest seen)
                if self.expected_max_hp is not None:
                    max_hp = min(max_hp, self.expected_max_hp)
                self.expected_max_hp = max_hp

                # Current HP should never increase during combat
                if self.in_combat and self.hp_history:
                    last_current = self.hp_history[-1] if self.hp_history else current
                    # If current is way higher than last, it's probably OCR error
                    if current > last_current * 1.1:  # Allow 10% tolerance
                        current = int(last_current * 0.95)  # Estimate slight decrease

                percent = round((current / max_hp) * 100, 1)
                return {"current": current, "max": max_hp, "percent": percent, "text": f"{current}/{max_hp}"}

            return None
        except Exception as e:
            print(f"OCR error: {e}")
            return None

    def _validate_hp_reading(self, current: int, max_hp: int) -> bool:
        """Validate OCR HP reading for sanity."""
        # Current can't exceed max
        if current > max_hp:
            return False

        # Max HP should be reasonable (camps have 1000-6000 HP typically)
        if max_hp < 500 or max_hp > 10000:
            return False

        # If we have an expected max HP, new reading should be close
        if self.expected_max_hp is not None:
            # Allow 10% variance in max HP reading
            if abs(max_hp - self.expected_max_hp) > self.expected_max_hp * 0.15:
                return False
        else:
            # First reading - set expected max HP
            self.expected_max_hp = max_hp

        return True

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

    def attack_move_click(self, x: int, y: int):
        """A + left-click spam with cursor spasm for style and effectiveness."""
        if not self.can_act():
            return
        # Spam A-clicks with random offsets for spasm effect
        for _ in range(4):
            ox = random.randint(-30, 30)
            oy = random.randint(-30, 30)
            cx, cy = x + ox, y + oy
            subprocess.run(["/opt/homebrew/bin/cliclick", "t:a", f"c:{cx},{cy}"], check=True)
            time.sleep(0.03)
        self.last_action_time = time.time()
        print(f"  -> A-click spam at ({x}, {y})")

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
        # Reset expected max HP for new camp
        self.expected_max_hp = None

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
        kite_distance = 450  # pixels from center to click

        click_x = int(screen_center_x + dx * kite_distance)
        click_y = int(screen_center_y + dy * kite_distance)

        # Clamp to screen bounds
        click_x = max(50, min(SCREEN_WIDTH - 50, click_x))
        click_y = max(50, min(SCREEN_HEIGHT - 50, click_y))

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
        if self.in_combat:
            # Level up ability after this camp
            ability_to_level = LEVEL_UP_ORDER.get(self.camp_index)
            if ability_to_level:
                self.level_up_ability(ability_to_level)
            self.in_combat = False
            self.camp_selected = False
            self.initial_engage_done = False  # Reset for next camp
            self.reset_hp_history()  # Clear HP history for next camp

        self.camp_index += 1
        if self.camp_index < len(CLEAR_ORDER):
            self.target_camp = CLEAR_ORDER[self.camp_index]
            self.state = State.WALKING_TO_CAMP
            print(f"Next camp: {self.target_camp}")
        else:
            self.state = State.IDLE
            print("Clear complete!")

    def _grok_worker(self):
        """Background thread for Grok API calls."""
        print("[GROK THREAD] Started")
        while self.grok_running:
            try:
                # Wait for frame data (with timeout so we can check grok_running)
                frame_data = self.grok_queue.get(timeout=0.5)
            except Empty:
                continue

            if frame_data is None:  # Shutdown signal
                break

            try:
                # Unpack frame data
                som_img = frame_data["som_img"]
                camp_health = frame_data["camp_health"]
                state_name = frame_data["state_name"]
                target_camp = frame_data["target_camp"]
                detections = frame_data["detections"]
                current_zone = frame_data["current_zone"]
                game_time = frame_data["game_time"]
                is_planner = frame_data.get("is_planner", False)

                if is_planner:
                    # Deep planner call
                    plan = self.strategist.plan(
                        img=som_img,
                        recent_logs=self.logs[-10:],
                        current_state=state_name,
                        target_camp=target_camp
                    )
                    plan_short = plan[:80] + "..." if len(plan) > 80 else plan
                    print(f"[GROK THREAD] PLANNER: {plan_short}")
                    # Store for main thread to display (Qt must update from main thread)
                    with self.grok_decision_lock:
                        self.last_grok_message = f"[PLAN] {plan_short}"
                else:
                    # Advisory analysis (Grok observes, doesn't control)
                    raw_response = self.strategist.decide(
                        img=som_img,
                        ocr_health=camp_health,
                        current_state=state_name,
                        target_camp=target_camp,
                        detections=detections,
                        current_zone=current_zone,
                        game_time=game_time
                    )

                    if raw_response:
                        analysis = raw_response.get('analysis', '')
                        issue = raw_response.get('issue')
                        tip = raw_response.get('tip')

                        # Build display message
                        lines = [f"[ANALYSIS] {analysis}"]
                        if issue:
                            lines.append(f"[ISSUE] {issue}")
                        if tip:
                            lines.append(f"[TIP] {tip}")

                        grok_msg = "\n".join(lines)
                        print(f"[GROK] {analysis}")

                        with self.grok_decision_lock:
                            self.last_grok_message = grok_msg

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
        W_APPROACH_DISTANCE = 25  # Minimap pixels - activate W when this close
        if self.camp_index >= 1 and mm_pos:  # After blue, we have W leveled
            dist = self.distance_to_camp(mm_pos, self.target_camp)
            if dist < W_APPROACH_DISTANCE:
                self.use_w_if_ready()

        # --- QUEUE FRAME FOR GROK (advisory/logging only) ---
        if self.strategist:
            self.planner_counter += 1
            is_planner = (self.planner_counter % PLANNER_INTERVAL == 0) and self.planner_counter > 0

            # Only queue occasionally to reduce load
            if self.frame_count % 5 == 0 or is_planner:
                som_img = draw_som_overlay(img, detections)
                frame_data = {
                    "som_img": som_img,
                    "camp_health": camp_health,
                    "state_name": self.state.name,
                    "target_camp": self.target_camp,
                    "detections": detections,
                    "current_zone": current_zone,
                    "game_time": game_time,
                    "is_planner": is_planner,
                }

                if self.grok_queue.full():
                    try:
                        self.grok_queue.get_nowait()
                    except Empty:
                        pass
                try:
                    self.grok_queue.put_nowait(frame_data)
                except:
                    pass

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
