#!/usr/bin/env python3
"""Always-on-top terminal overlay for bot status."""

import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette
from collections import deque


class OverlayWindow(QWidget):
    """Small always-on-top terminal overlay."""

    def __init__(self, max_lines: int = 15):
        super().__init__()
        self.max_lines = max_lines
        self.lines = deque(maxlen=max_lines)
        self.setup_ui()

    def setup_ui(self):
        # Window flags: always on top, no frame, stays on all workspaces
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.X11BypassWindowManagerHint  # Helps with focus issues
        )

        # Make window semi-transparent and not take focus
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)

        # Size and position (bottom left)
        self.setGeometry(10, 800, 500, 280)  # x, y, width, height

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

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


# Global overlay instance
_overlay = None
_app = None


def init_overlay():
    """Initialize the overlay (call from main thread before bot starts)."""
    global _overlay, _app
    if _overlay is None:
        _app = QApplication.instance() or QApplication(sys.argv)
        _overlay = OverlayWindow()
    return _overlay


def log_overlay(message: str):
    """Log a message to the overlay."""
    global _overlay
    if _overlay:
        _overlay.log(message)


def process_events():
    """Process Qt events (call periodically from bot loop)."""
    global _app
    if _app:
        _app.processEvents()


if __name__ == "__main__":
    # Test the overlay
    app = QApplication(sys.argv)
    overlay = OverlayWindow()

    # Simulate log messages
    import time
    timer = QTimer()
    counter = [0]

    def add_log():
        counter[0] += 1
        overlay.log(f"[{counter[0]}] GROK: ATTACK -> blue_buff | Camp visible, engaging")
        if counter[0] >= 20:
            timer.stop()

    timer.timeout.connect(add_log)
    timer.start(500)

    sys.exit(app.exec())
