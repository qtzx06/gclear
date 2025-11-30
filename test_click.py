#!/usr/bin/env python3
"""Test if cliclick works."""

import time
import subprocess

def click(x: int, y: int, right: bool = False):
    """Click using cliclick."""
    cmd = "rc" if right else "c"
    subprocess.run(["cliclick", f"{cmd}:{x},{y}"], check=True)
    print(f"Clicked at ({x}, {y})")

if __name__ == "__main__":
    print("Right-clicking at (500, 500) in 2 seconds...")
    time.sleep(2)
    click(500, 500, right=True)
    print("Done!")
