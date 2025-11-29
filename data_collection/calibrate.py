#!/usr/bin/env python3
"""
Calibration tool for ROIs and minimap zones.
- Drag ROIs to reposition
- Click minimap to define zones
- Outputs pixel coordinates to update config
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QFrame
)
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QCursor

from config import SCREEN_WIDTH, SCREEN_HEIGHT, ROIS, ROI_COLORS


class DraggableROI:
    def __init__(self, name, x, y, w, h, color):
        self.name = name
        self.rect = QRect(x, y, w, h)
        self.color = color
        self.dragging = False
        self.resizing = False
        self.drag_offset = QPoint()

    def contains(self, pos):
        return self.rect.contains(pos)

    def corner_rect(self):
        """Bottom-right corner for resizing"""
        return QRect(self.rect.right() - 15, self.rect.bottom() - 15, 15, 15)

    def in_corner(self, pos):
        return self.corner_rect().contains(pos)


class CalibrationOverlay(QMainWindow):
    def __init__(self):
        super().__init__()
        self.rois = []
        self.zones = {}  # zone_name -> (x1, y1, x2, y2)
        self.current_zone = None
        self.zone_start = None
        self.defining_zone = False
        self.selected_roi = None
        self.mouse_pos = QPoint()

        self.init_rois()
        self.init_ui()

    def init_rois(self):
        for name, (x, y, w, h) in ROIS.items():
            r, g, b, a = ROI_COLORS.get(name, (255, 255, 255, 180))
            self.rois.append(DraggableROI(name, x, y, w, h, QColor(r, g, b, a)))

    def init_ui(self):
        self.setWindowTitle("gclear - Calibration")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
        self.setMouseTracking(True)
        self.show()

        # Control panel
        self.control = ControlPanel(self)
        self.control.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw ROIs
        for roi in self.rois:
            # Fill
            fill = QColor(roi.color)
            fill.setAlpha(40)
            painter.fillRect(roi.rect, fill)

            # Border
            pen = QPen(roi.color)
            pen.setWidth(3 if roi == self.selected_roi else 2)
            painter.setPen(pen)
            painter.drawRect(roi.rect)

            # Resize handle
            painter.fillRect(roi.corner_rect(), roi.color)

            # Label
            painter.setPen(QColor(255, 255, 255, 230))
            painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            painter.drawText(roi.rect.x() + 5, roi.rect.y() + 18, roi.name)

            # Coordinates
            painter.setFont(QFont("Arial", 9))
            coord_text = f"({roi.rect.x()}, {roi.rect.y()}, {roi.rect.width()}, {roi.rect.height()})"
            painter.drawText(roi.rect.x() + 5, roi.rect.bottom() - 5, coord_text)

        # Draw minimap zones (if minimap ROI exists)
        minimap_roi = next((r for r in self.rois if r.name == "minimap"), None)
        if minimap_roi:
            for zone_name, (x1, y1, x2, y2) in self.zones.items():
                # Convert relative to absolute
                abs_x1 = minimap_roi.rect.x() + x1
                abs_y1 = minimap_roi.rect.y() + y1
                abs_x2 = minimap_roi.rect.x() + x2
                abs_y2 = minimap_roi.rect.y() + y2

                zone_rect = QRect(abs_x1, abs_y1, abs_x2 - abs_x1, abs_y2 - abs_y1)
                painter.setPen(QPen(QColor(0, 255, 0, 200), 2))
                painter.drawRect(zone_rect)
                painter.drawText(abs_x1 + 2, abs_y1 + 12, zone_name)

            # Drawing current zone
            if self.defining_zone and self.zone_start:
                painter.setPen(QPen(QColor(255, 255, 0, 200), 2, Qt.PenStyle.DashLine))
                current_rect = QRect(self.zone_start, self.mouse_pos).normalized()
                painter.drawRect(current_rect)

        # Mouse coordinates
        painter.setPen(QColor(255, 255, 255, 200))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(self.mouse_pos.x() + 15, self.mouse_pos.y() - 5,
                        f"({self.mouse_pos.x()}, {self.mouse_pos.y()})")

        painter.end()

    def mousePressEvent(self, event):
        pos = event.pos()

        if self.defining_zone:
            # Start defining zone on minimap
            minimap_roi = next((r for r in self.rois if r.name == "minimap"), None)
            if minimap_roi and minimap_roi.rect.contains(pos):
                self.zone_start = pos
            return

        # Check for ROI interaction
        for roi in reversed(self.rois):  # Top to bottom
            if roi.in_corner(pos):
                roi.resizing = True
                self.selected_roi = roi
                return
            elif roi.contains(pos):
                roi.dragging = True
                roi.drag_offset = pos - roi.rect.topLeft()
                self.selected_roi = roi
                return

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()

        for roi in self.rois:
            if roi.dragging:
                new_pos = event.pos() - roi.drag_offset
                roi.rect.moveTopLeft(new_pos)
                self.update()
                self.control.update_roi_list()
                return
            elif roi.resizing:
                new_w = max(50, event.pos().x() - roi.rect.x())
                new_h = max(30, event.pos().y() - roi.rect.y())
                roi.rect.setWidth(new_w)
                roi.rect.setHeight(new_h)
                self.update()
                self.control.update_roi_list()
                return

        self.update()

    def mouseReleaseEvent(self, event):
        if self.defining_zone and self.zone_start and self.current_zone:
            minimap_roi = next((r for r in self.rois if r.name == "minimap"), None)
            if minimap_roi:
                # Convert to relative coordinates within minimap
                x1 = self.zone_start.x() - minimap_roi.rect.x()
                y1 = self.zone_start.y() - minimap_roi.rect.y()
                x2 = event.pos().x() - minimap_roi.rect.x()
                y2 = event.pos().y() - minimap_roi.rect.y()

                # Normalize
                self.zones[self.current_zone] = (
                    min(x1, x2), min(y1, y2),
                    max(x1, x2), max(y1, y2)
                )
                self.control.update_zone_list()

            self.zone_start = None
            self.defining_zone = False
            self.control.zone_btn.setText("Add Zone")
            self.update()
            return

        for roi in self.rois:
            roi.dragging = False
            roi.resizing = False

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.defining_zone = False
            self.zone_start = None
            self.control.zone_btn.setText("Add Zone")
            self.update()


class ControlPanel(QMainWindow):
    def __init__(self, overlay):
        super().__init__()
        self.overlay = overlay
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Calibration Controls")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        self.setGeometry(SCREEN_WIDTH + 10, 100, 350, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Instructions
        instructions = QLabel("Drag ROIs to reposition\nDrag corners to resize")
        instructions.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(instructions)

        # ROI List
        layout.addWidget(QLabel("ROIs:"))
        self.roi_list = QListWidget()
        self.roi_list.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.roi_list)
        self.update_roi_list()

        # Zone controls
        layout.addWidget(QLabel("Minimap Zones:"))

        zone_row = QHBoxLayout()
        self.zone_input = QLabel("Click 'Add Zone' then draw on minimap")
        self.zone_input.setStyleSheet("color: #666; font-size: 10px;")
        zone_row.addWidget(self.zone_input)
        layout.addLayout(zone_row)

        zone_btns = QHBoxLayout()
        self.zone_btn = QPushButton("Add Zone")
        self.zone_btn.clicked.connect(self.start_zone_definition)
        zone_btns.addWidget(self.zone_btn)

        self.zone_name_btns = []
        for name in ["blue_buff", "gromp", "wolves", "raptors", "red_buff", "krugs", "spawn"]:
            btn = QPushButton(name[:4])
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda checked, n=name: self.set_zone_name(n))
            zone_btns.addWidget(btn)
            self.zone_name_btns.append(btn)
        layout.addLayout(zone_btns)

        self.zone_list = QListWidget()
        self.zone_list.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.zone_list.setMaximumHeight(150)
        layout.addWidget(self.zone_list)

        # Export button
        export_btn = QPushButton("Export Config")
        export_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        export_btn.clicked.connect(self.export_config)
        layout.addWidget(export_btn)

        # Output
        layout.addWidget(QLabel("Config Output:"))
        self.output = QLabel("(click Export)")
        self.output.setStyleSheet("font-family: monospace; font-size: 10px; background: #1a1a1a; color: #0f0; padding: 10px;")
        self.output.setWordWrap(True)
        self.output.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.output)

        layout.addStretch()

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close_all)
        layout.addWidget(close_btn)

    def update_roi_list(self):
        self.roi_list.clear()
        for roi in self.overlay.rois:
            r = roi.rect
            text = f"{roi.name}: ({r.x()}, {r.y()}, {r.width()}, {r.height()})"
            self.roi_list.addItem(text)

    def update_zone_list(self):
        self.zone_list.clear()
        for name, coords in self.overlay.zones.items():
            text = f"{name}: {coords}"
            self.zone_list.addItem(text)

    def set_zone_name(self, name):
        self.overlay.current_zone = name
        self.zone_btn.setText(f"Drawing: {name}")

    def start_zone_definition(self):
        if self.overlay.current_zone:
            self.overlay.defining_zone = True
            self.zone_btn.setText(f"Draw {self.overlay.current_zone} on minimap...")
        else:
            self.zone_btn.setText("Select zone name first!")

    def export_config(self):
        lines = ["# Updated ROIs"]
        lines.append("ROIS = {")
        for roi in self.overlay.rois:
            r = roi.rect
            lines.append(f'    "{roi.name}": ({r.x()}, {r.y()}, {r.width()}, {r.height()}),')
        lines.append("}")
        lines.append("")
        lines.append("# Minimap zones (relative to minimap ROI)")
        lines.append("MINIMAP_ZONES = {")
        for name, coords in self.overlay.zones.items():
            lines.append(f'    "{name}": {coords},')
        lines.append("}")

        output = "\n".join(lines)
        self.output.setText(output)
        print(output)

    def close_all(self):
        self.overlay.close()
        self.close()


def main():
    print("Calibration tool")
    print("- Drag ROIs to reposition")
    print("- Drag corners to resize")
    print("- Select zone name, click 'Add Zone', draw on minimap")
    print("- Export config when done")

    app = QApplication(sys.argv)
    overlay = CalibrationOverlay()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
