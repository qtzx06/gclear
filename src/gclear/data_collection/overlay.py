#!/usr/bin/env python3
"""
League of Legends Data Collection Overlay
- Transparent overlay showing ROI regions
- Control panel for capturing screenshots
- Extracts and previews ROI regions
"""

import sys
import os
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGridLayout, QFrame
)
from PyQt6.QtCore import Qt, QRect, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap, QImage, QShortcut, QKeySequence

import Quartz
from Quartz import CGWindowListCreateImage, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
from PIL import Image

from config import SCREEN_WIDTH, SCREEN_HEIGHT, ROIS, ROI_COLORS, DATA_DIR


class OverlayWindow(QMainWindow):
    """Transparent overlay that shows ROI regions on screen."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Frameless, transparent, always on top, click-through
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Cover the game region
        self.setGeometry(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for name, (x, y, w, h) in ROIS.items():
            r, g, b, a = ROI_COLORS.get(name, (255, 255, 255, 150))

            # Draw filled rectangle with transparency
            fill_color = QColor(r, g, b, 40)
            painter.fillRect(x, y, w, h, fill_color)

            # Draw border
            pen = QPen(QColor(r, g, b, a))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(x, y, w, h)

            # Draw label
            painter.setPen(QColor(255, 255, 255, 200))
            painter.drawText(x + 5, y + 15, name)

        painter.end()


class ControlPanel(QMainWindow):
    """Control panel for capturing screenshots and viewing ROI previews."""

    def __init__(self, overlay: OverlayWindow):
        super().__init__()
        self.overlay = overlay
        self.session_dir = None
        self.capture_count = 0
        self.init_ui()
        self.start_session()

    def init_ui(self):
        self.setWindowTitle("gclear - Data Collection")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )

        # Position to the right of the game
        self.setGeometry(SCREEN_WIDTH + 10, 100, 400, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Status label
        self.status_label = QLabel("Session: starting...")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.status_label)

        # Capture count
        self.count_label = QLabel("Captures: 0")
        self.count_label.setStyleSheet("font-size: 12px; color: #666;")
        layout.addWidget(self.count_label)

        # Capture button
        self.capture_btn = QPushButton("CAPTURE (Space)")
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 20px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.capture_btn.clicked.connect(self.capture)
        layout.addWidget(self.capture_btn)

        # Toggle overlay button
        self.toggle_btn = QPushButton("Hide Overlay")
        self.toggle_btn.clicked.connect(self.toggle_overlay)
        layout.addWidget(self.toggle_btn)

        # ROI Previews
        layout.addWidget(QLabel("ROI Previews:"))

        preview_frame = QFrame()
        preview_frame.setStyleSheet("background-color: #1a1a1a; border-radius: 4px;")
        preview_layout = QGridLayout(preview_frame)

        self.preview_labels = {}
        for i, name in enumerate(ROIS.keys()):
            container = QVBoxLayout()

            label = QLabel(name)
            label.setStyleSheet("color: white; font-size: 10px;")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container.addWidget(label)

            preview = QLabel()
            preview.setFixedSize(180, 100)
            preview.setStyleSheet("background-color: #333; border: 1px solid #555;")
            preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            preview.setText("No capture")
            preview.setStyleSheet("color: #666; background-color: #333;")
            self.preview_labels[name] = preview
            container.addWidget(preview)

            wrapper = QWidget()
            wrapper.setLayout(container)
            preview_layout.addWidget(wrapper, i // 2, i % 2)

        layout.addWidget(preview_frame)
        layout.addStretch()

        # Keyboard shortcut - use QShortcut for better reliability
        self.shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.shortcut.activated.connect(self.capture)

        self.show()

    def start_session(self):
        """Create a new session directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(DATA_DIR) / timestamp
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirs for ROIs
        (self.session_dir / "full").mkdir(exist_ok=True)
        for name in ROIS.keys():
            (self.session_dir / name).mkdir(exist_ok=True)

        self.status_label.setText(f"Session: {timestamp}")
        self.capture_count = 0
        self.count_label.setText("Captures: 0")

    def capture_screen(self) -> Image.Image:
        """Capture the screen using macOS Quartz."""
        # Hide overlay temporarily for clean capture
        self.overlay.hide()
        QApplication.processEvents()

        # Small delay to ensure overlay is hidden
        import time
        time.sleep(0.05)

        # Capture screen
        region = Quartz.CGRectMake(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        image_ref = CGWindowListCreateImage(
            region,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
            0
        )

        if image_ref is None:
            self.overlay.show()
            return None

        # Convert to PIL Image
        width = Quartz.CGImageGetWidth(image_ref)
        height = Quartz.CGImageGetHeight(image_ref)
        bytes_per_row = Quartz.CGImageGetBytesPerRow(image_ref)

        data_provider = Quartz.CGImageGetDataProvider(image_ref)
        data = Quartz.CGDataProviderCopyData(data_provider)

        img = Image.frombytes("RGBA", (width, height), data, "raw", "BGRA", bytes_per_row, 1)

        # Show overlay again
        self.overlay.show()

        return img

    def capture(self):
        """Capture screenshot and extract ROIs."""
        img = self.capture_screen()
        if img is None:
            self.status_label.setText("Capture failed!")
            return

        self.capture_count += 1
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]

        # Save full screenshot
        full_path = self.session_dir / "full" / f"{timestamp}.png"
        img.save(full_path)

        # Extract and save ROIs
        for name, (x, y, w, h) in ROIS.items():
            roi_img = img.crop((x, y, x + w, y + h))
            roi_path = self.session_dir / name / f"{timestamp}.png"
            roi_img.save(roi_path)

            # Update preview
            self.update_preview(name, roi_img)

        self.count_label.setText(f"Captures: {self.capture_count}")

        # Flash the button green
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #8BC34A;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 20px;
                border-radius: 8px;
            }
        """)
        QTimer.singleShot(100, self.reset_button_style)

    def reset_button_style(self):
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 20px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)

    def update_preview(self, name: str, pil_img: Image.Image):
        """Update the preview label with a PIL image."""
        # Convert PIL to QPixmap
        pil_img = pil_img.convert("RGB")
        data = pil_img.tobytes()
        bytes_per_line = pil_img.width * 3
        qimg = QImage(data, pil_img.width, pil_img.height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())  # copy() ensures data stays valid

        # Scale to fit preview
        label = self.preview_labels[name]
        scaled = pixmap.scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled)

    def toggle_overlay(self):
        """Toggle overlay visibility."""
        if self.overlay.isVisible():
            self.overlay.hide()
            self.toggle_btn.setText("Show Overlay")
        else:
            self.overlay.show()
            self.toggle_btn.setText("Hide Overlay")

    def closeEvent(self, event):
        """Clean up on close."""
        self.overlay.close()
        event.accept()


def main():
    # Check for screen recording permission
    print("Starting gclear data collection overlay...")
    print(f"Screen size: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print(f"ROIs: {list(ROIS.keys())}")
    print(f"Data will be saved to: {DATA_DIR}/")
    print()
    print("NOTE: You may need to grant Screen Recording permission to Terminal/Python")
    print("      System Preferences > Privacy & Security > Screen Recording")
    print()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    overlay = OverlayWindow()
    control = ControlPanel(overlay)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
