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
from overlay import init_overlay, log_overlay, log_strategist, strategist_thinking, strategist_stream, strategist_result, tick_strategist, process_events
from som import draw_som_overlay, get_region

DEBUG_VIEW = False  # Show detection window (blocks control)
DEBUG_RECORD = True  # Save debug frames
RUNS_DIR = "runs"

# Strategist config
USE_STRATEGIST = True  # Grok runs continuously to verify camp clears


class State(Enum):
    # Early game states
    STARTUP_LEVELUP = auto()    # Level up Q at game start
    STARTUP_BUY = auto()        # Buy starter item
    WALKING_TO_CAMP = auto()    # Walking via minimap
    WAITING_FOR_SPAWN = auto()  # At camp, waiting for 1:30

    # Combat states
    ENGAGING = auto()           # Moving to attack camp
    ATTACKING = auto()          # In combat, spamming abilities

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
Q_COOLDOWN = 0.3  # Spam Q every 0.3 sec during combat
W_COOLDOWN = 12.73

# Game timing constants
CAMP_SPAWN_TIME = 90  # 1:30 in seconds
EARLY_GAME_END = 25   # First 25 seconds for buying

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

# Attack config
ATTACK_MOVE_INTERVAL = 0.3  # Seconds between A-clicks
HP_SMOOTHING_FRAMES = 3  # Average HP over this many frames


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
        self.last_attack_move_time = 0  # Track attack-move timing
        self.camp_selected = False  # Track if we've left-clicked to show health bar
        self.initial_engage_done = False  # Track if initial attack_click done
        self.hp_history = []  # Track recent HP readings
        self.no_detection_frames = 0  # Count frames without camp detection (for robust clear detection)
        self.last_known_hp = None  # Last known HP reading for fallback
        self.camp_engage_time = None  # When we started fighting current camp
        self.grok_verified_clear = False  # Grok confirmed camp is dead
        self.combat_frames = 0  # Count frames in combat for Grok timing
        self.last_grok_request_frame = 0  # Track when we last asked Grok
        self.zero_hp_frames = 0  # Count consecutive frames at 0 HP

        # Early game tracking
        self.leveled_up_q = False
        self.bought_item = False
        self.pressed_hotkey = False  # Ctrl+Shift+O after buying

        # Strategist (threaded)
        self.strategist = JungleStrategist() if USE_STRATEGIST else None
        self.last_grok_decision = None
        self.last_grok_message = None  # For main thread to display
        self.grok_decision_lock = threading.Lock()
        self.grok_thread = None
        self.grok_running = False
        self.latest_frame_data = None  # Shared data for Grok to grab
        self.frame_data_lock = threading.Lock()

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

    def get_zone_probabilities(self, mm_pos: tuple[int, int]) -> dict:
        """Calculate probability of being at each camp based on distance."""
        if not mm_pos:
            return {}

        # Calculate distances to all camps
        distances = {}
        for camp in CLEAR_ORDER:
            distances[camp] = self.distance_to_camp(mm_pos, camp)

        # Convert to probabilities (inverse distance, softmax-ish)
        # Closer = higher probability
        min_dist = min(distances.values()) or 1
        probs = {}
        total = 0
        for camp, dist in distances.items():
            # Use inverse distance squared for sharper falloff
            score = 1.0 / ((dist / min_dist) ** 2 + 0.1)
            probs[camp] = score
            total += score

        # Normalize
        for camp in probs:
            probs[camp] = round(probs[camp] / total * 100, 1)

        return probs

    def get_map_side(self, mm_pos: tuple[int, int]) -> str:
        """Determine if player is on TOP side or BOT side of jungle."""
        if not mm_pos:
            return "unknown"

        probs = self.get_zone_probabilities(mm_pos)
        top_side = ["blue_buff", "gromp", "wolves"]
        bot_side = ["raptors", "red_buff", "krugs"]

        top_prob = sum(probs.get(c, 0) for c in top_side)
        bot_prob = sum(probs.get(c, 0) for c in bot_side)

        if top_prob > bot_prob:
            return f"TOP_SIDE ({top_prob:.0f}%)"
        else:
            return f"BOT_SIDE ({bot_prob:.0f}%)"

    def compute_state(
        self,
        game_time_str: str,
        current_zone: str,
        target_camp: str,
        camp_detection: Optional[Detection],
        camp_health: Optional[dict],
        mm_pos: Optional[tuple]
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

        # Multi-monster camps (wolves, raptors) - MUST stay for minimum time
        multi_monster_camps = {"wolves", "raptors"}
        min_fight_time = 17.0 if target_camp == "wolves" else 15.0  # wolves gets 2 extra sec
        close_distance = 13  # minimap pixels - must be this close to count as "at camp"

        if target_camp in multi_monster_camps:
            try:
                # Check if we're actually close to the camp
                dist_to_camp = self.distance_to_camp(mm_pos, target_camp) if mm_pos else 999
                is_close_to_camp = dist_to_camp <= close_distance

                # Track when we got close to the camp
                if self.camp_engage_time is None and is_close_to_camp:
                    self.camp_engage_time = time.time()
                    log_overlay(f"At {target_camp} (dist={dist_to_camp:.0f}) - 15s timer started")

                # If timer started and we're close, force stay for minimum time
                if self.camp_engage_time is not None and is_close_to_camp:
                    time_at_camp = time.time() - self.camp_engage_time
                    if time_at_camp < min_fight_time:
                        # Not enough time yet - FORCE attacking no matter what
                        return State.ATTACKING

                    # After 15 sec, only move on if Grok says dead
                    if self.grok_verified_clear:
                        log_overlay(f"{target_camp} - Grok verified after {min_fight_time}s - CLEARED")
                        return State.CAMP_CLEARED
                    else:
                        # Still waiting for Grok - keep attacking
                        return State.ATTACKING
                else:
                    # Not close enough - need to walk there
                    return State.WALKING_TO_CAMP
            except Exception as e:
                print(f"Error in multi-monster camp logic: {e}")
                return State.ATTACKING  # Default to attacking on error

        else:
            # Regular camps - use detection + Grok/HP logic
            camp_still_detected = camp_detection is not None

            if camp_still_detected:
                # Monster still on screen - keep fighting, reset counters
                self.no_detection_frames = 0
                self.zero_hp_frames = 0
            else:
                # Monster not detected - count frames
                self.no_detection_frames += 1

                # Track 0 HP frames (only when not detected)
                if camp_health and camp_health.get("current", 100) <= 0:
                    self.zero_hp_frames += 1
                else:
                    self.zero_hp_frames = 0

                # Move on if: not detected for a few frames AND (Grok says dead OR 0 HP for a while)
                if self.no_detection_frames >= 3:
                    if self.grok_verified_clear:
                        log_overlay(f"{target_camp} not detected + Grok verified - CLEARED")
                        return State.CAMP_CLEARED
                    elif self.zero_hp_frames >= 10:
                        log_overlay(f"{target_camp} not detected + 0 HP for 10 frames - CLEARED")
                        return State.CAMP_CLEARED

        # Check distance to target camp
        dist_to_camp = self.distance_to_camp(mm_pos, target_camp) if mm_pos else 999
        is_close = dist_to_camp <= 15  # Must be within 15 minimap pixels

        # Is target camp visible on screen?
        if camp_detection and is_close:
            self.no_detection_frames = 0  # Reset counter
            # Do we have it targeted (health bar showing)?
            if camp_health:
                # Track when we started fighting
                if self.camp_engage_time is None:
                    self.camp_engage_time = time.time()
                return State.ATTACKING
            else:
                # Camp visible but not targeted - engage it
                return State.ENGAGING
        elif is_close:
            # Close but camp not visible - count frames
            self.no_detection_frames += 1

            # We're close but camp not detected - keep attacking or engaging
            if self.in_combat:
                return State.ATTACKING
            else:
                return State.ENGAGING
        else:
            # Not close enough - need to walk there
            return State.WALKING_TO_CAMP

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
            self.levelup_time = time.time()  # Track when we leveled up
            log_overlay("Leveled up Q")

        elif state == State.STARTUP_BUY:
            # Wait 1 second after leveling Q before buying
            if hasattr(self, 'levelup_time') and time.time() - self.levelup_time < 1.0:
                return  # Wait
            # Buy starter item
            self._buy_starter_item()
            self.bought_item = True
            log_overlay("Bought starter item")

        elif state == State.WALKING_TO_CAMP:
            # Level up W when walking to gromp (after blue)
            if self.target_camp == "gromp" and self.camp_index == 1:
                self.level_up_ability("W")

            # Click minimap to walk to target camp - spam it
            region_id = f"mm_{self.target_camp.replace('_buff', '')}"
            region = get_region(region_id)
            if region:
                # Spam right click on minimap
                for _ in range(3):
                    subprocess.run(["/opt/homebrew/bin/cliclick", f"rc:{region.x},{region.y}"], check=True)
                    time.sleep(0.05)
                log_overlay(f"Walking to {self.target_camp}")
            else:
                log_overlay(f"No region found: {region_id}")

        elif state == State.WAITING_FOR_SPAWN:
            # Auto attack at 1:29 and 1:35 to prep for camp spawn
            game_secs = parse_game_time(self.ocr_timer(self.capture_screen()) or "0:00")
            if game_secs in (89, 95):  # 1:29 and 1:35
                # Find player and attack near them
                player_pos = None
                for d in detections:
                    if d.cls == "player":
                        player_pos = (d.x, d.y)
                        break
                if player_pos:
                    subprocess.run(["/opt/homebrew/bin/cliclick", "t:a", f"c:{player_pos[0]},{player_pos[1]}"], check=True)
                    log_overlay(f"Pre-attack at {game_secs}s")

        elif state == State.ENGAGING:
            # Camp visible - left click to show HP bar, then A + left click to attack
            if camp_detection and self.can_act():
                x, y = camp_detection.x, camp_detection.y
                # Left click to select (shows HP bar)
                subprocess.run(["/opt/homebrew/bin/cliclick", f"c:{x},{y}"], check=True)
                time.sleep(0.05)
                # A + left click to attack move
                subprocess.run(["/opt/homebrew/bin/cliclick", "t:a", f"c:{x},{y}"], check=True)
                time.sleep(0.05)
                # Another left click for good measure
                subprocess.run(["/opt/homebrew/bin/cliclick", f"c:{x},{y}"], check=True)
                self.camp_selected = True
                self.last_action_time = time.time()
                log_overlay(f"Engaging {self.target_camp}")

        elif state == State.ATTACKING:
            # In combat - spam attacks and abilities
            # Find player position from detections
            player_pos = None
            for d in detections:
                if d.cls == "player":
                    player_pos = (d.x, d.y)
                    break

            # Get attack position - use camp if detected, otherwise player position
            if camp_detection:
                attack_x, attack_y = camp_detection.x, camp_detection.y
            elif player_pos:
                # No camp detected - attack near player position
                attack_x, attack_y = player_pos
            else:
                # Fallback to screen center
                attack_x, attack_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

            if not self.initial_engage_done:
                # Initial attack
                use_spam = self.target_camp in SPAM_CLICK_CAMPS
                self.attack_click(attack_x, attack_y, spam=use_spam)
                self.initial_engage_done = True
                self.in_combat = True
            else:
                # Continue attacking - spam left click + A click
                subprocess.run(["/opt/homebrew/bin/cliclick", f"c:{attack_x},{attack_y}"], check=True)
                time.sleep(0.02)
                self.attack_move_click(attack_x, attack_y)

            # Spam Q always
            self.use_q_if_ready()
            self.use_w_if_ready()

            # Spam smite at blue buff and krugs
            if self.target_camp in ("blue_buff", "krugs"):
                self.press_key("f")

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
        if self.in_combat:
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
        """A + left-click on the camp/target position."""
        if not self.can_act():
            return

        # Click on camp position with small random offset
        for _ in range(3):
            ox = random.randint(-20, 20)
            oy = random.randint(-20, 20)
            cx, cy = x + ox, y + oy
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
        """Spam Q constantly."""
        self.press_ability("Q")

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
            time.sleep(1.0)  # Wait for shop to fully open

            # Gustwalker Hatchling in recommended items - single right click to buy
            item_pos = (650, 340)
            print(f"  -> Right-clicking item at {item_pos}...")
            subprocess.run(["/opt/homebrew/bin/cliclick", f"rc:{item_pos[0]},{item_pos[1]}"], check=True)
            time.sleep(0.5)

            # Close shop with B
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
        self.no_detection_frames = 0
        self.last_known_hp = None
        self.camp_engage_time = None
        self.grok_verified_clear = False
        self.combat_frames = 0
        self.last_grok_request_frame = 0
        self.zero_hp_frames = 0
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
        """Background thread for Grok - runs continuously on its own loop."""
        print("[GROK] Starting continuous loop...")

        # Set up streaming callback
        def on_stream(chunk, full_text):
            strategist_stream(chunk, full_text)

        self.strategist.set_stream_callback(on_stream)

        # Show loading state
        with self.grok_decision_lock:
            self.last_grok_message = "[GROK] waiting..."

        while self.grok_running:
            try:
                # Grab latest frame data
                with self.frame_data_lock:
                    frame_data = self.latest_frame_data

                if frame_data is None:
                    # No data yet
                    with self.grok_decision_lock:
                        self.last_grok_message = "[GROK] waiting for game..."
                    time.sleep(0.5)
                    continue

                # Get game time for display
                game_t = frame_data.get("game_time", "?")

                # Start thinking animation (will be animated in main thread)
                strategist_thinking()
                with self.grok_decision_lock:
                    self.last_grok_message = f"[...] @{game_t}"

                # Call Grok (streams via callback)
                result = self.strategist.verify_camp_cleared(
                    img=frame_data["img"],
                    target_camp=frame_data["target_camp"],
                    time_at_camp=frame_data["time_at_camp"],
                    last_hp_percent=frame_data["last_hp_percent"],
                    detections=frame_data["detections"],
                    current_zone=frame_data["current_zone"],
                    game_time=frame_data["game_time"],
                    zone_probs=frame_data.get("zone_probs"),
                    map_side=frame_data.get("map_side", "unknown")
                )

                camp_dead = result.get("camp_dead", False)
                confidence = result.get("confidence", 0.0)
                reason = result.get("reason", "")

                # Build message with game time only
                status = "DEAD" if camp_dead else "ALIVE"
                target = frame_data['target_camp']
                grok_msg = f"@{game_t} {target}: {status} ({confidence:.0%})"
                grok_msg_full = f"{grok_msg}\n{reason}"
                print(f"[GROK] {grok_msg} - {reason}")

                # Show final result
                strategist_result(grok_msg_full)

                with self.grok_decision_lock:
                    self.last_grok_message = grok_msg_full
                    if camp_dead and confidence >= 0.7:
                        self.grok_verified_clear = True
                        print(f"[GROK] VERIFIED CLEAR: {target}")

                # Wait 1 second after response before next call
                time.sleep(1.0)

            except Exception as e:
                print(f"[GROK] Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

        print("[GROK] Stopped")

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
            camp_health=camp_health,
            mm_pos=mm_pos
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

        # --- UPDATE GROK FRAME DATA ---
        # Always update latest frame data for Grok's continuous loop
        if self.strategist:
            if self.in_combat:
                self.combat_frames += 1
            else:
                self.combat_frames = 0

            # Calculate time fighting this camp
            time_at_camp = 0.0
            if self.camp_engage_time:
                time_at_camp = time.time() - self.camp_engage_time

            # Get last known HP percent
            last_hp_pct = None
            if self.last_known_hp:
                last_hp_pct = self.last_known_hp.get("percent")

            # Calculate zone probabilities and map side
            zone_probs = self.get_zone_probabilities(mm_pos) if mm_pos else {}
            map_side = self.get_map_side(mm_pos) if mm_pos else "unknown"

            frame_data = {
                "img": img.copy(),  # Copy so Grok thread has stable data
                "target_camp": self.target_camp,
                "time_at_camp": time_at_camp,
                "last_hp_percent": last_hp_pct,
                "detections": detections.copy(),
                "current_zone": current_zone,
                "game_time": game_time,
                "zone_probs": zone_probs,
                "map_side": map_side,
            }

            with self.frame_data_lock:
                self.latest_frame_data = frame_data

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
                try:
                    self.update()
                except Exception as e:
                    print(f"ERROR in update: {e}")
                    import traceback
                    traceback.print_exc()
                    with open('gclear_crash.log', 'a') as f:
                        f.write(f"\n--- Frame {self.frame_count} ---\n")
                        f.write(traceback.format_exc())
                    log_overlay(f"ERROR: {e}")
                tick_strategist()  # Animate thinking indicator
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
    import logging

    # Set up file logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('gclear_debug.log', mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('gclear')

    try:
        logger.info("Starting GClear bot...")
        bot = JungleBot()
        bot.run()
    except Exception as e:
        logger.exception(f"CRASH: {e}")
        import traceback
        with open('gclear_crash.log', 'w') as f:
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
