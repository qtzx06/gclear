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

# Minimum game time (seconds) before camp can possibly be cleared
# blue spawns at 1:30, each camp takes ~8-10s to clear
MIN_GAME_TIME_FOR_CLEAR = {
    "blue_buff": 90 + 8,     # 1:38 earliest (spawn 1:30 + ~8s clear)
    "gromp": 90 + 14,        # 1:44 (after blue)
    "wolves": 90 + 20,       # 1:50 (after gromp)
    "raptors": 90 + 28,      # 1:58 (after wolves + walk)
    "red_buff": 90 + 36,     # 2:06 (after raptors)
    "krugs": 90 + 46,        # 2:16 (after red)
}


def parse_game_time(game_time: str) -> Optional[int]:
    """Parse game time string (e.g. '1:30' or '2:05') to seconds."""
    if not game_time or game_time == "?":
        return None
    try:
        parts = game_time.split(":")
        if len(parts) == 2:
            mins, secs = int(parts[0]), int(parts[1])
            return mins * 60 + secs
    except:
        pass
    return None


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


def extract_verdict_from_response(response: str) -> dict:
    """Extract verdict from JSON response."""
    # Try to parse JSON
    try:
        result = extract_json_from_response(response)
        if result and "dead" in result:
            return {
                "camp_dead": result["dead"],
                "confidence": 0.9,
                "reason": result.get("reason", "")
            }
    except:
        pass

    # Fallback - look for keywords
    response_lower = response.lower()
    if "dead" in response_lower and "not" not in response_lower:
        return {"camp_dead": True, "confidence": 0.7, "reason": "Inferred dead"}
    if "alive" in response_lower or "still" in response_lower:
        return {"camp_dead": False, "confidence": 0.7, "reason": "Inferred alive"}

    return None


def build_camp_verify_prompt(
    target_camp: str,
    time_at_camp: float,
    last_hp_percent: Optional[float],
    detections: list,
    current_zone: Optional[str],
    game_time: Optional[str],
    zone_probs: dict = None,
    map_side: str = "unknown",
    is_early_game: bool = False
) -> str:
    """Build prompt for camp clear verification with detailed analysis."""

    min_time = MIN_KILL_TIME.get(target_camp, 6)
    max_hp = CAMP_MAX_HP.get(target_camp, 2000)
    next_camp_idx = CLEAR_ORDER.index(target_camp) + 1 if target_camp in CLEAR_ORDER else -1
    next_camp = CLEAR_ORDER[next_camp_idx] if next_camp_idx < len(CLEAR_ORDER) else "DONE"

    det_str = ", ".join([f"{d.cls}" for d in detections]) if detections else "NONE"

    # Format zone probabilities
    if zone_probs:
        top_3 = sorted(zone_probs.items(), key=lambda x: -x[1])[:3]
        prob_str = ", ".join([f"{c}:{p:.0f}%" for c, p in top_3])
    else:
        prob_str = "unknown"

    if is_early_game:
        return f"""Is camp dead? Respond ONLY with JSON: {{"dead": false, "reason": "waiting for spawn"}}"""

    return f"""Look at screenshot. Is {target_camp} dead? Check if monsters still visible.
Respond ONLY with JSON: {{"dead": true/false, "reason": "brief reason"}}"""


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
        self.stream_callback = None  # Called with each chunk of text

    def set_stream_callback(self, callback):
        """Set callback for streaming text updates."""
        self.stream_callback = callback

    def _call_grok_stream(self, img: Image.Image, prompt: str) -> Optional[str]:
        """Call Grok vision API with streaming and chain-of-thought reasoning."""
        image_data = encode_image_to_base64(img)

        payload = {
            "model": "grok-4-1-fast-reasoning",
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
            "temperature": 0.7,
            "stream": True
        }

        try:
            full_content = ""
            full_reasoning = ""
            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", API_URL, json=payload, headers=self.headers) as response:
                    response.raise_for_status()
                    buffer = ""
                    for chunk in response.iter_text():
                        buffer += chunk
                        # Process complete lines
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line or line == "data: [DONE]":
                                continue
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    choices = data.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        # Check for reasoning content (chain of thought)
                                        reasoning = delta.get("reasoning_content", "")
                                        if reasoning:
                                            full_reasoning += reasoning
                                            if self.stream_callback:
                                                self.stream_callback(reasoning, full_reasoning)
                                        # Regular content
                                        content = delta.get("content", "")
                                        if content:
                                            full_content += content
                                            if self.stream_callback:
                                                self.stream_callback(content, full_reasoning + "\n---\n" + full_content)
                                except json.JSONDecodeError:
                                    pass

            # Return both reasoning and content
            if full_reasoning:
                return full_reasoning + "\n---\n" + full_content
            return full_content
        except Exception as e:
            print(f"Grok API error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def verify_camp_cleared(
        self,
        img: Image.Image,
        target_camp: str,
        time_at_camp: float,
        last_hp_percent: Optional[float],
        detections: list,
        current_zone: Optional[str],
        game_time: Optional[str] = None,
        zone_probs: dict = None,
        map_side: str = "unknown"
    ) -> dict:
        """
        Ask Grok: Is this camp dead? Should we move to next?

        Returns: {"camp_dead": bool, "confidence": float, "reason": str}
        """
        # Check if game time is early (before 1:30) - tell Grok to plan instead
        game_secs = parse_game_time(game_time)
        is_early_game = game_secs is not None and game_secs < 90

        prompt = build_camp_verify_prompt(
            target_camp=target_camp,
            time_at_camp=time_at_camp,
            last_hp_percent=last_hp_percent if last_hp_percent else 100.0,
            detections=detections,
            current_zone=current_zone,
            game_time=game_time,
            zone_probs=zone_probs,
            map_side=map_side,
            is_early_game=is_early_game
        )

        response = self._call_grok_stream(img, prompt)
        if not response:
            # API failed - be conservative, assume camp NOT dead
            return {"camp_dead": False, "confidence": 0.0, "reason": "API error"}

        # Try to extract verdict from response
        result = extract_verdict_from_response(response)
        if result:
            self.last_result = result
            return result

        # Fallback to JSON extraction
        result = extract_json_from_response(response)
        if result and "camp_dead" in result:
            self.last_result = result
            return result

        # Couldn't parse response - be conservative
        return {"camp_dead": False, "confidence": 0.5, "reason": "No clear verdict"}

    def close(self):
        """Clean up."""
        pass
