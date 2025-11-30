#!/usr/bin/env python3
"""Grok-powered camp clear verifier - ONLY decides if camp is dead and should move."""

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

CLEAR_ORDER = ["blue_buff", "gromp", "wolves", "raptors", "red_buff", "krugs"]

# Expected time to kill each camp (seconds) - if less than this, probably not dead
MIN_KILL_TIME = {
    "blue_buff": 8,
    "gromp": 5,
    "wolves": 6,
    "raptors": 5,
    "red_buff": 8,
    "krugs": 7,
}

# Expected max HP for each camp
CAMP_MAX_HP = {
    "blue_buff": 2300,
    "gromp": 2050,
    "wolves": 1600,
    "raptors": 1200,
    "red_buff": 2300,
    "krugs": 1400,
}


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def encode_image_to_base64(img: Image.Image, max_size: int = 800) -> str:
    """Encode PIL image to base64 JPEG."""
    w, h = img.size
    if w > max_size or h > max_size:
        ratio = min(max_size / w, max_size / h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


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


def build_camp_verify_prompt(
    target_camp: str,
    time_at_camp: float,
    last_hp_percent: Optional[float],
    detections: list,
    current_zone: Optional[str],
    game_time: Optional[str]
) -> str:
    """Build prompt for camp clear verification - ONLY decides if camp is dead."""

    min_time = MIN_KILL_TIME.get(target_camp, 6)
    max_hp = CAMP_MAX_HP.get(target_camp, 2000)
    next_camp_idx = CLEAR_ORDER.index(target_camp) + 1 if target_camp in CLEAR_ORDER else -1
    next_camp = CLEAR_ORDER[next_camp_idx] if next_camp_idx < len(CLEAR_ORDER) else "DONE"

    det_str = ", ".join([f"{d.cls}" for d in detections]) if detections else "NONE"

    return f"""You are a jungle clear verifier. Your ONLY job: decide if {target_camp} is DEAD and should move to {next_camp}.

DATA:
- Target camp: {target_camp} (max HP: {max_hp})
- Time fighting this camp: {time_at_camp:.1f} seconds
- Minimum expected kill time: {min_time} seconds
- Last known HP: {last_hp_percent:.0f}% (if available)
- Current detections on screen: {det_str}
- Player zone (minimap): {current_zone or "unknown"}
- Game time: {game_time or "unknown"}

CLEAR ORDER: blue_buff -> gromp -> wolves -> raptors -> red_buff -> krugs

RULES:
1. If time_at_camp < min_kill_time, camp is probably NOT dead (detection error)
2. If HP was recently high (>30%), camp is probably NOT dead
3. If target camp class is still detected on screen, camp is NOT dead
4. Only say camp_dead=true if you're CONFIDENT it's actually dead

Look at the screenshot. Is {target_camp} actually dead?

Respond with ONLY this JSON:
{{"camp_dead": true/false, "confidence": 0.0-1.0, "reason": "<brief reason>"}}"""


# -----------------------------------------------------------------------------
# Main strategist class
# -----------------------------------------------------------------------------

class JungleStrategist:
    """Grok vision - ONLY verifies if camp is dead and should move to next."""

    def __init__(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        self.last_result = None

    def _call_grok(self, img: Image.Image, prompt: str) -> Optional[str]:
        """Call Grok vision API with reasoning model."""
        image_data = encode_image_to_base64(img)

        payload = {
            "model": "grok-4-1-fast-reasoning",  # Use reasoning for better judgment
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": 0,
            "stream": False
        }

        try:
            response = httpx.post(API_URL, json=payload, headers=self.headers, timeout=30.0)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Grok API error: {e}")
            return None

    def verify_camp_cleared(
        self,
        img: Image.Image,
        target_camp: str,
        time_at_camp: float,
        last_hp_percent: Optional[float],
        detections: list,
        current_zone: Optional[str],
        game_time: Optional[str] = None
    ) -> dict:
        """
        Ask Grok: Is this camp dead? Should we move to next?

        Returns: {"camp_dead": bool, "confidence": float, "reason": str}
        """
        prompt = build_camp_verify_prompt(
            target_camp=target_camp,
            time_at_camp=time_at_camp,
            last_hp_percent=last_hp_percent if last_hp_percent else 100.0,
            detections=detections,
            current_zone=current_zone,
            game_time=game_time
        )

        response = self._call_grok(img, prompt)
        if not response:
            # API failed - be conservative, assume camp NOT dead
            return {"camp_dead": False, "confidence": 0.0, "reason": "API error - assuming not dead"}

        result = extract_json_from_response(response)
        if result and "camp_dead" in result:
            self.last_result = result
            return result

        # Couldn't parse response - be conservative
        return {"camp_dead": False, "confidence": 0.0, "reason": "Parse error - assuming not dead"}

    def close(self):
        """Clean up."""
        pass
