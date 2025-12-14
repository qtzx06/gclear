#!/usr/bin/env python3
"""Test Grok vision API with game screenshot."""

import os
import base64
import httpx
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

API_KEY = os.getenv("XAI_API_KEY")
API_URL = "https://api.x.ai/v1/chat/completions"


def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def call_grok_vision(image_path: str, prompt: str, model: str = "grok-4-1-fast-non-reasoning") -> str:
    """Call Grok vision API with an image."""
    image_data = encode_image(image_path)

    # Determine image type
    ext = Path(image_path).suffix.lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
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

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = httpx.post(API_URL, json=payload, headers=headers, timeout=30.0)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def check_health_reading(image_path: str) -> dict:
    """Test health bar reading."""
    prompt = """Look at this game screenshot health bar.
    Return ONLY a JSON object with the health values:
    {"current": <number>, "max": <number>, "percent": <number>}

    If you can't read the health bar, return: {"error": "cannot read"}
    """

    response = call_grok_vision(image_path, prompt)
    print(f"Health response: {response}")
    return response


def check_strategy(image_path: str) -> str:
    """Test strategic analysis with reasoning model."""
    prompt = """You are a League of Legends jungle clearing assistant for Hecarim.

    Analyze this game screenshot and determine:
    1. Current situation (what camp, player position, health status)
    2. What action to take next

    Respond with a brief JSON:
    {
        "situation": "<brief description>",
        "action": "<ATTACK|KITE|WALK|WAIT>",
        "target": "<camp name or direction>",
        "reasoning": "<why>"
    }
    """

    response = call_grok_vision(image_path, prompt, model="grok-4-1-fast-reasoning")
    print(f"Strategy response: {response}")
    return response


if __name__ == "__main__":
    # Find a test frame
    runs_dir = Path("runs")
    if runs_dir.exists():
        # Find most recent run with frames
        runs = sorted(runs_dir.iterdir(), reverse=True)
        for run in runs:
            frames_dir = run / "frames"
            if frames_dir.exists():
                frames = list(frames_dir.glob("*.png"))
                if frames:
                    test_frame = frames[0]
                    print(f"Testing with: {test_frame}")

                    print("\n=== Testing Health Reading ===")
                    check_health_reading(str(test_frame))

                    print("\n=== Testing Strategy ===")
                    check_strategy(str(test_frame))
                    break
        else:
            print("No frames found in runs/")
    else:
        print("No runs/ directory found")
