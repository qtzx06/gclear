#!/usr/bin/env python3
"""Grok-powered jungle clearing strategist with full control."""

import os
import base64
import json
import httpx
from typing import Optional
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from som import get_som_prompt_section

load_dotenv()

API_KEY = os.getenv("XAI_API_KEY")
API_URL = "https://api.x.ai/v1/chat/completions"


# -----------------------------------------------------------------------------
# Screen layout constants
# -----------------------------------------------------------------------------

SCREEN_INFO = """
SCREEN LAYOUT (1710x1107):
- Game view: Center of screen, Hecarim (armored centaur) is the player
- Health bar ROI: top left shows targeted camp HP as "current/max"
- Minimap: bottom right corner

MINIMAP CLICK POSITIONS (for WALK action):
- blue_buff: (1440, 838)
- gromp: (1408, 829)
- wolves: (1435, 875)
- raptors: (1511, 905)
- red_buff: (1531, 938)
- krugs: (1544, 965)
"""

CAMP_DESCRIPTIONS = """
=== CAMP VISUAL DESCRIPTIONS ===
YOLO detects these as class names. Here's what each looks like:

"blue" = BLUE SENTINEL (Blue Buff):
  - Stone golem with GLOWING BLUE ORB floating above its head
  - Rocky gray body, humanoid shape
  - Has 2300 max HP

"gromp" = GROMP:
  - Large brown FROG/TOAD creature
  - Sits on a rock, has big eyes
  - Bulky, wide body
  - Has 2100 max HP

"wolves" = WOLVES:
  - Pack of gray/white wolves
  - One large wolf + smaller wolves
  - Has ~1800 max HP (big wolf)

"raptors" = RAPTORS:
  - Red/orange chicken-like birds
  - One large raptor + small ones
  - Has ~1200 max HP (big raptor)

"red" = RED BRAMBLEBACK (Red Buff):
  - Large creature with GLOWING RED stones on back
  - Brown/red coloring, hunched posture
  - Has 2300 max HP

"krugs" = KRUGS:
  - Rock creatures, stone golems
  - One large + medium + small krugs
  - Gray/brown rocks

"player" = HECARIM (you):
  - Armored centaur (horse body, human torso)
  - Blue/teal glowing weapon
  - This is YOU - don't attack this!

"mm_player" = Player icon on minimap (small blue dot)
"""

ABILITIES_INFO = """
ABILITIES:
- Q: Main damage spell - USE FREQUENTLY in combat
- W: Heal/sustain
- F: Smite - USE when camp HP < 500 to secure kill

LEVEL UP (use "level_up" field):
- Set "level_up": "Q" to level up Q ability (Ctrl+Q)
- Level Q at game start (before 0:15)
"""

STARTUP_INFO = """
=== STARTUP SEQUENCE (game time < 1:30) ===
If game_time shows early game (< 0:15):
1. LEVEL_UP Q first (set "level_up": "Q")
2. SHOP action to buy Gustwalker Hatchling
3. WALK to blue_buff position

If game_time is between 0:15 and 1:30:
- WALK to blue_buff and WAIT (camp spawns at 1:30)

If game_time >= 1:30:
- Blue buff has spawned, start clearing!
"""

CLEAR_ORDER = ["blue_buff", "gromp", "wolves", "raptors", "red_buff", "krugs"]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def encode_image_to_base64(img: Image.Image, max_size: int = 800) -> str:
    """Encode PIL image to base64 JPEG (smaller = faster upload)."""
    # Resize if too large
    w, h = img.size
    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=70)  # JPEG is much smaller than PNG
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def format_detections(detections: list) -> str:
    """Format detection list for prompt."""
    if not detections:
        return "NONE - screen may be empty or camp dead"
    return ", ".join([f"{d.cls}@({d.x},{d.y})" for d in detections])


def format_health(ocr_health: Optional[dict]) -> str:
    """Format health info for prompt."""
    if not ocr_health:
        return "NONE (no health bar visible - use SELECT to focus camp, or camp is dead)"
    return f"VISIBLE: {ocr_health['current']}/{ocr_health['max']} = {ocr_health['percent']}%"


def extract_json_from_response(response: str) -> Optional[dict]:
    """Extract JSON object from response text."""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass
    return None


def build_tactical_prompt(
    detection_str: str,
    health_str: str,
    current_state: str,
    target_camp: str,
    current_zone: Optional[str],
    planner_context: str,
    game_time: Optional[str] = None
) -> str:
    """Build advisory prompt - Grok analyzes, state machine acts."""
    return f"""Hecarim jungle clear advisor. Analyze the screenshot.

Current: state={current_state}, target={target_camp}, zone={current_zone or "?"}, time={game_time or "?"}
Detections: {detection_str}
HP: {health_str}

Brief analysis (1-2 sentences): What's happening? Any issues?

JSON: {{"analysis":"<brief>","issue":"<problem or null>","tip":"<advice or null>"}}"""


def build_planner_prompt(
    log_str: str,
    current_state: str,
    target_camp: str
) -> str:
    """Build the strategic planner prompt."""
    return f"""You are the strategic planner for Hecarim jungle clear.

{SCREEN_INFO}

CURRENT SITUATION:
- State: {current_state}
- Target: {target_camp}

RECENT HISTORY:
{log_str}

CLEAR ORDER: blue_buff -> gromp -> wolves -> raptors -> red_buff -> krugs

Analyze the screenshot and history. Think about:
1. Are we on track with the clear?
2. Any problems or inefficiencies?
3. What should we prepare for?

Provide a BRIEF strategic directive (2-3 sentences max) that will guide tactical decisions.
Focus on: timing, positioning, ability usage, when to kite vs full clear."""


# -----------------------------------------------------------------------------
# Main strategist class
# -----------------------------------------------------------------------------

class JungleStrategist:
    """Uses Grok vision to decide jungle clearing actions with full control."""

    def __init__(self):
        # Note: Don't use persistent httpx.Client - not thread-safe
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        self.last_decision = None
        self.planner_context = ""
        self.decision_count = 0

    def _call_grok(self, img: Image.Image, prompt: str, reasoning: bool = False) -> Optional[str]:
        """Call Grok vision API (thread-safe)."""
        image_data = encode_image_to_base64(img)
        model = "grok-4-1-fast-reasoning" if reasoning else "grok-4-1-fast-non-reasoning"

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "temperature": 0,
            "stream": False
        }

        try:
            # Use httpx.post directly (thread-safe, creates new connection)
            response = httpx.post(API_URL, json=payload, headers=self.headers, timeout=30.0)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Grok API error: {e}")
            return None

    def decide(
        self,
        img: Image.Image,
        ocr_health: Optional[dict],
        current_state: str,
        target_camp: str,
        detections: list,
        current_zone: Optional[str],
        game_time: Optional[str] = None
    ) -> dict:
        """Get tactical decision from Grok (every frame)."""
        detection_str = format_detections(detections)
        health_str = format_health(ocr_health)

        prompt = build_tactical_prompt(
            detection_str=detection_str,
            health_str=health_str,
            current_state=current_state,
            target_camp=target_camp,
            current_zone=current_zone,
            planner_context=self.planner_context,
            game_time=game_time
        )

        response = self._call_grok(img, prompt, reasoning=False)
        if not response:
            return self._fallback(current_state, target_camp, ocr_health)

        decision = extract_json_from_response(response)
        if decision:
            self.last_decision = decision
            self.decision_count += 1
            return decision

        return self._fallback(current_state, target_camp, ocr_health)

    def plan(
        self,
        img: Image.Image,
        recent_logs: list,
        current_state: str,
        target_camp: str
    ) -> str:
        """Deep reasoning planner (slower, every 10 frames)."""
        # Summarize recent logs
        log_summary = []
        for log in recent_logs[-5:]:
            grok_decision = log.get('grok_decision') or {}
            reasoning = grok_decision.get('reasoning', 'N/A')
            log_summary.append(f"- {log.get('state')} @ {log.get('zone')}: {reasoning}")
        log_str = "\n".join(log_summary) if log_summary else "No history"

        prompt = build_planner_prompt(log_str, current_state, target_camp)

        response = self._call_grok(img, prompt, reasoning=True)
        if response:
            # Extract just the key directive
            self.planner_context = response[:200] if len(response) > 200 else response
            return self.planner_context

        return self.planner_context or "Continue standard clear."

    def _fallback(self, current_state: str, target_camp: str, ocr_health: Optional[dict]) -> dict:
        """Fallback decision when API fails."""
        # If no health bar, try to select first
        if ocr_health is None:
            action = "SELECT"
        elif current_state == "ATTACK_CAMP":
            action = "ATTACK"
        else:
            action = "WALK"

        return {
            "action": action,
            "target": target_camp,
            "click": None,
            "ability": "Q" if action == "ATTACK" else None,
            "health_estimate": ocr_health["percent"] if ocr_health else None,
            "reasoning": "API fallback"
        }

    def close(self):
        """Clean up."""
        pass  # No persistent client to close
