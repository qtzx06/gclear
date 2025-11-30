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

load_dotenv()

API_KEY = os.getenv("XAI_API_KEY")
API_URL = "https://api.x.ai/v1/chat/completions"


# -----------------------------------------------------------------------------
# Screen layout constants
# -----------------------------------------------------------------------------

SCREEN_INFO = """
SCREEN LAYOUT (1710x1107):
- Game view: Center of screen, player character usually near center
- Health bar ROI: (132, 72, 104, 36) - top left area shows targeted camp HP
- Timer ROI: (1648, 64, 50, 32) - top right shows game time
- Minimap: (1336, 658, 370, 360) - bottom right corner

MINIMAP CLICK POSITIONS (for walking to camps):
- blue_buff: (1440, 838)
- gromp: (1408, 829)
- wolves: (1439, 870)
- raptors: (1511, 893)
- red_buff: (1531, 924)
- krugs: (1544, 950)

ABILITIES:
- Q: Main damage spell (use often during combat)
- W: Heal/sustain
- E: Speed boost charge
- F: Smite (use on big camps when HP low, ~300-500)

ATTACK CONTROLS:
- Right-click: Move to location / attack target
- Left-click: SELECT target (shows health bar above it)
- A + Left-click: Attack-move (auto-attacks nearest enemy)
"""

CLEAR_ORDER = ["blue_buff", "gromp", "wolves", "raptors", "red_buff", "krugs"]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def encode_image_to_base64(img: Image.Image) -> str:
    """Encode PIL image to base64 PNG."""
    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True)
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
    planner_context: str
) -> str:
    """Build the tactical decision prompt."""
    return f"""You control Hecarim jungle clear. You receive DATA from our systems - TRUST THIS DATA over screenshot analysis.

{SCREEN_INFO}

=== SYSTEM DATA (TRUST THIS) ===
DETECTIONS (from CV model): {detection_str}
HEALTH BAR (from OCR): {health_str}
PLAYER ZONE: {current_zone or "unknown"}
TARGET CAMP: {target_camp}
BOT STATE: {current_state}

STRATEGIC CONTEXT: {planner_context or "Start blue buff clear"}

CLEAR ORDER: blue_buff -> gromp -> wolves -> raptors -> red_buff -> krugs

=== DECISION LOGIC ===
Use the SYSTEM DATA above to decide:

1. If HEALTH BAR says "VISIBLE: X/Y = Z%" -> Camp is targeted, you can ATTACK
2. If HEALTH BAR says "NONE" but DETECTIONS show camp -> Use SELECT to target it
3. If DETECTIONS show no camp AND HEALTH is NONE -> Camp dead, use FINISH
4. If HEALTH shows < 40% -> Consider KITE toward next camp
5. If HEALTH shows < 500 HP -> Use SMITE
6. If camp not in DETECTIONS -> Use WALK to minimap position

=== ACTIONS ===
- ATTACK: Right-click + A-click camp. Use when health VISIBLE.
- SELECT: Left-click camp to target it. Use when health NONE but camp detected.
- KITE: Move toward next camp while attacking. Use when HP < 40%.
- WALK: Click minimap to walk to camp. Use when camp not detected.
- SMITE: Press F. Use when HP ~300-500.
- FINISH: Camp dead, advance to next. Use when no camp AND no health.
- WAIT: Do nothing this frame.

=== ABILITIES ===
- "Q": Main damage - USE EVERY ATTACK FRAME
- "W": Heal
- "F": Smite (finisher)

RESPOND JSON ONLY:
{{
  "action": "ATTACK" | "SELECT" | "KITE" | "WALK" | "SMITE" | "FINISH" | "WAIT",
  "target": "{target_camp}",
  "click": {{"x": <int>, "y": <int>}} or null,
  "ability": "Q" | "W" | "F" | null,
  "reasoning": "<brief>"
}}"""


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
        self.client = httpx.Client(timeout=20.0)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        self.last_decision = None
        self.planner_context = ""
        self.decision_count = 0

    def _call_grok(self, img: Image.Image, prompt: str, reasoning: bool = False) -> Optional[str]:
        """Call Grok vision API."""
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
                                "url": f"data:image/png;base64,{image_data}"
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
            response = self.client.post(API_URL, json=payload, headers=self.headers)
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
        current_zone: Optional[str]
    ) -> dict:
        """Get tactical decision from Grok (fast, every 2 frames)."""
        detection_str = format_detections(detections)
        health_str = format_health(ocr_health)

        prompt = build_tactical_prompt(
            detection_str=detection_str,
            health_str=health_str,
            current_state=current_state,
            target_camp=target_camp,
            current_zone=current_zone,
            planner_context=self.planner_context
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
        self.client.close()
