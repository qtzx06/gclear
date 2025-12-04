#!/usr/bin/env python3
"""Always-on-top terminal overlays for bot status and strategist."""

import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextOption
from collections import deque


class OverlayWindow(QWidget):
    """Minimal always-on-top terminal overlay."""

    def __init__(self, max_lines: int = 8, x: int = 10, y: int = 800, width: int = 350, height: int = 140, title: str = ""):
        super().__init__()
        self.max_lines = max_lines
        self.lines = deque(maxlen=max_lines)
        self.pos_x = x
        self.pos_y = y
        self.width_px = width
        self.height_px = height
        self.title = title
        self.setup_ui()

    def setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setGeometry(self.pos_x, self.pos_y, self.width_px, self.height_px)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Monaco", 9))
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: rgba(15, 15, 15, 230);
                color: #e0e0e0;
                border: 1px solid rgba(80, 80, 80, 150);
                border-radius: 4px;
                padding: 4px 6px 4px 12px;
            }
        """)

        layout.addWidget(self.text_edit)

        if self.title:
            self.lines.append(f"--- {self.title} ---")
            self.text_edit.setPlainText("\n".join(self.lines))

        self.show()

    def log(self, message: str):
        """Add a log message."""
        self.lines.append(message)
        self.text_edit.setPlainText("\n".join(self.lines))
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        """Clear all messages."""
        self.lines.clear()
        self.text_edit.clear()


class GrokOverlay(QWidget):
    """Grok overlay with streaming text display."""

    def __init__(self, x: int = 50, y: int = 820, width: int = 240, height: int = 200):
        super().__init__()
        self.pos_x = x
        self.pos_y = y
        self.width_px = width
        self.height_px = height
        self._spinner_idx = 0
        self._thinking = False
        self._stream_text = ""
        self._last_result = ""
        self.setup_ui()

    def setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setGeometry(self.pos_x, self.pos_y, self.width_px, self.height_px)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Monaco", 8))
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)

        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: rgba(15, 15, 15, 230);
                color: #a0a0a0;
                border: 1px solid rgba(80, 80, 80, 150);
                border-radius: 4px;
                padding: 6px 8px;
            }
        """)

        layout.addWidget(self.text_edit)
        self._render()
        self.show()

    def _render(self):
        """Render the overlay."""
        lines = []
        lines.append("─── GROK ───")

        if self._thinking:
            dots = "." * ((self._spinner_idx % 3) + 1)
            dots_pad = " " * (3 - len(dots))
            lines.append(f"Thinking{dots}{dots_pad}")
            lines.append("")
            # Show streaming text (truncate to fit)
            if self._stream_text:
                # Word wrap and limit lines
                text = self._stream_text[-400:]  # Last 400 chars
                lines.append(text)
        else:
            lines.append("")
            if self._last_result:
                lines.append(self._last_result)

        self.text_edit.setPlainText("\n".join(lines))
        # Scroll to bottom
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_thinking(self):
        """Start thinking animation."""
        self._thinking = True
        self._spinner_idx = 0
        self._stream_text = ""
        self._render()

    def tick(self):
        """Animate the thinking dots."""
        if self._thinking:
            self._spinner_idx += 1
            self._render()

    def _clean_text(self, text: str) -> str:
        """Clean up Grok text - remove markdown headers, make it readable."""
        import re
        # Replace ### headers with bold-style text
        text = re.sub(r'^###\s*(.+)$', r'▸ \1', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s*(.+)$', r'▸ \1', text, flags=re.MULTILINE)
        text = re.sub(r'^#\s*(.+)$', r'▸ \1', text, flags=re.MULTILINE)
        # Remove ** bold markers
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        # Remove * italic markers
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        # Clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)

    def stream_text(self, chunk: str, full_text: str):
        """Update with streamed text."""
        self._stream_text = self._clean_text(full_text)
        self._render()

    def set_result(self, message: str):
        """Set final result (stops thinking)."""
        self._thinking = False
        self._last_result = self._clean_text(message)
        self._stream_text = ""
        self._render()


# Global instances
_status_overlay = None
_grok_overlay = None
_app = None

# Thread-safe state
_grok_state = {
    "thinking": False,
    "result": None,
    "stream_chunk": None,
    "stream_full": None,
}


def init_overlay():
    """Initialize overlays."""
    global _status_overlay, _grok_overlay, _app
    if _status_overlay is None:
        _app = QApplication.instance() or QApplication(sys.argv)
        _status_overlay = OverlayWindow(max_lines=12, x=100, y=540, width=240, height=200, title="BOT")
        _grok_overlay = GrokOverlay(x=100, y=745, width=240, height=200)
    return _status_overlay


def log_overlay(message: str):
    """Log to bot overlay."""
    global _status_overlay
    if _status_overlay:
        _status_overlay.log(message)


def log_strategist(message: str):
    """Log to grok overlay (sets result)."""
    global _grok_overlay
    if _grok_overlay:
        _grok_overlay.set_result(message)


def strategist_thinking():
    """Signal thinking started (thread-safe)."""
    global _grok_state
    _grok_state["thinking"] = True
    _grok_state["result"] = None
    _grok_state["stream_chunk"] = None
    _grok_state["stream_full"] = None


def strategist_stream(chunk: str, full_text: str):
    """Signal streaming text (thread-safe)."""
    global _grok_state
    _grok_state["stream_chunk"] = chunk
    _grok_state["stream_full"] = full_text


def strategist_result(message: str):
    """Signal result ready (thread-safe)."""
    global _grok_state
    _grok_state["thinking"] = False
    _grok_state["result"] = message


def tick_strategist():
    """Update grok overlay from main thread."""
    global _grok_overlay, _grok_state
    if not _grok_overlay:
        return

    # Check for result first
    if _grok_state["result"]:
        _grok_overlay.set_result(_grok_state["result"])
        _grok_state["result"] = None
    elif _grok_state["thinking"]:
        if not _grok_overlay._thinking:
            _grok_overlay.start_thinking()
        else:
            _grok_overlay.tick()
        # Check for stream updates
        if _grok_state["stream_full"]:
            _grok_overlay.stream_text(_grok_state["stream_chunk"], _grok_state["stream_full"])
            _grok_state["stream_chunk"] = None


def process_events():
    """Process Qt events."""
    global _app
    if _app:
        _app.processEvents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    status = OverlayWindow(max_lines=12, x=100, y=540, width=240, height=200, title="BOT")
    grok = GrokOverlay(x=100, y=745, width=240, height=200)

    timer = QTimer()
    counter = [0]
    stream_text = [""]

    def update():
        counter[0] += 1
        status.log(f"[{counter[0]}] Zone:blue | Target:blue")

        if counter[0] < 20:
            if counter[0] == 1:
                grok.start_thinking()
            grok.tick()
            # Simulate streaming
            stream_text[0] += "analyzing image... checking camp HP... "
            grok.stream_text("", stream_text[0])
        elif counter[0] == 20:
            grok.set_result("@1:42 blue: ALIVE (90%)")
            stream_text[0] = ""
        elif counter[0] < 40:
            if counter[0] == 25:
                grok.start_thinking()
            grok.tick()
            stream_text[0] += "HP low, checking if dead... "
            grok.stream_text("", stream_text[0])
        elif counter[0] == 40:
            grok.set_result("@1:52 blue: DEAD (95%)")

        if counter[0] >= 50:
            timer.stop()

    timer.timeout.connect(update)
    timer.start(100)

    sys.exit(app.exec())
