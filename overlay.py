#!/usr/bin/env python3
"""Always-on-top terminal overlays for bot status and strategist."""

import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette
from collections import deque


class OverlayWindow(QWidget):
    """Small always-on-top terminal overlay."""

    def __init__(self, max_lines: int = 15, x: int = 10, y: int = 800, title: str = ""):
        super().__init__()
        self.max_lines = max_lines
        self.lines = deque(maxlen=max_lines)
        self.pos_x = x
        self.pos_y = y
        self.title = title
        self.setup_ui()

    def setup_ui(self):
        # Window flags: always on top, no frame, stays on all workspaces
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.X11BypassWindowManagerHint
        )

        # Make window semi-transparent and not take focus
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)

        # Size and position
        self.setGeometry(self.pos_x, self.pos_y, 400, 250)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # Title label
        if self.title:
            self.title_label = QLabel(self.title)
            self.title_label.setFont(QFont("Monaco", 9))
            self.title_label.setStyleSheet("color: #888888; padding: 2px;")
            layout.addWidget(self.title_label)

        # Text display
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Monaco", 10))

        # Dark semi-transparent background with white text
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: rgba(20, 20, 20, 220);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 80);
                border-radius: 5px;
                padding: 5px;
            }
        """)

        layout.addWidget(self.text_edit)

        self.show()

    def log(self, message: str):
        """Add a log message."""
        self.lines.append(message)
        self.text_edit.setPlainText("\n".join(self.lines))
        # Scroll to bottom
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        """Clear all messages."""
        self.lines.clear()
        self.text_edit.clear()


# Global overlay instances
_status_overlay = None
_strategist_overlay = None
_app = None


def init_overlay():
    """Initialize both overlays (call from main thread before bot starts)."""
    global _status_overlay, _strategist_overlay, _app
    if _status_overlay is None:
        _app = QApplication.instance() or QApplication(sys.argv)
        # Status overlay on the left
        _status_overlay = OverlayWindow(max_lines=12, x=10, y=800, title="BOT STATUS")
        # Strategist overlay on the right
        _strategist_overlay = OverlayWindow(max_lines=12, x=420, y=800, title="GROK STRATEGIST")
        _strategist_overlay.log("waiting for first frame...")
    return _status_overlay


def log_overlay(message: str):
    """Log a message to the status overlay."""
    global _status_overlay
    if _status_overlay:
        _status_overlay.log(message)


def log_strategist(message: str):
    """Log a message to the strategist overlay."""
    global _strategist_overlay
    if _strategist_overlay:
        _strategist_overlay.log(message)


def strategist_thinking():
    """Show thinking status in strategist overlay."""
    global _strategist_overlay
    if _strategist_overlay:
        _strategist_overlay.log("thinking...")


def process_events():
    """Process Qt events (call periodically from bot loop)."""
    global _app
    if _app:
        _app.processEvents()


if __name__ == "__main__":
    # Test both overlays
    app = QApplication(sys.argv)
    status = OverlayWindow(max_lines=12, x=10, y=800, title="BOT STATUS")
    strategist = OverlayWindow(max_lines=12, x=420, y=800, title="GROK STRATEGIST")

    timer = QTimer()
    counter = [0]

    def add_log():
        counter[0] += 1
        status.log(f"[{counter[0]}] Zone:blue_buff | Target:blue_buff | HP:1800/2300")
        if counter[0] % 2 == 0:
            strategist.log("thinking...")
        else:
            strategist.log(f"> ATTACK @(820,515) Q")
            strategist.log("  Blue buff visible, attacking")
        if counter[0] >= 20:
            timer.stop()

    timer.timeout.connect(add_log)
    timer.start(500)

    sys.exit(app.exec())
