#!/usr/bin/env python3
"""
Calibration tool for ROIs and minimap zones.
Shows a screenshot with draggable/resizable ROI boxes.
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QScrollArea, QFileDialog
)
from PyQt6.QtCore import Qt, QPoint, QRect, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPixmap, QImage

from config import SCREEN_WIDTH, SCREEN_HEIGHT, ROIS, ROI_COLORS


class CalibrationWidget(QWidget):
    """Widget showing screenshot with draggable ROIs"""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.pixmap = None
        self.scale = 0.5  # Scale down for display
        self.rois = {}
        self.zones = {}
        self.selected_roi = None
        self.dragging = False
        self.resizing = False
        self.drag_offset = QPoint()
        self.mouse_pos = QPoint()

        # Zone drawing
        self.drawing_zone = False
        self.current_zone_name = None
        self.zone_start = None

        self.init_rois()
        self.setMouseTracking(True)
        self.setMinimumSize(int(SCREEN_WIDTH * self.scale), int(SCREEN_HEIGHT * self.scale))

    def init_rois(self):
        for name, (x, y, w, h) in ROIS.items():
            r, g, b, a = ROI_COLORS.get(name, (255, 255, 255, 180))
            self.rois[name] = {
                'rect': QRect(x, y, w, h),
                'color': QColor(r, g, b, a)
            }

    def load_image(self, path):
        self.pixmap = QPixmap(path)
        self.update()

    def scaled_rect(self, rect):
        """Convert real coords to display coords"""
        return QRect(
            int(rect.x() * self.scale),
            int(rect.y() * self.scale),
            int(rect.width() * self.scale),
            int(rect.height() * self.scale)
        )

    def real_pos(self, pos):
        """Convert display coords to real coords"""
        return QPoint(int(pos.x() / self.scale), int(pos.y() / self.scale))

    def corner_rect(self, rect):
        """Resize handle for a rect"""
        s = int(20 * self.scale)
        return QRect(rect.right() - s, rect.bottom() - s, s, s)

    def paintEvent(self, event):
        painter = QPainter(self)

        # Draw background image
        if self.pixmap:
            scaled_pixmap = self.pixmap.scaled(
                int(SCREEN_WIDTH * self.scale),
                int(SCREEN_HEIGHT * self.scale),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            painter.drawPixmap(0, 0, scaled_pixmap)
        else:
            painter.fillRect(self.rect(), QColor(30, 30, 30))
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load a screenshot first")

        # Draw ROIs
        for name, roi in self.rois.items():
            rect = self.scaled_rect(roi['rect'])
            color = roi['color']

            # Fill
            fill = QColor(color)
            fill.setAlpha(60)
            painter.fillRect(rect, fill)

            # Border
            pen = QPen(color)
            pen.setWidth(3 if name == self.selected_roi else 2)
            painter.setPen(pen)
            painter.drawRect(rect)

            # Resize handle
            corner = self.corner_rect(rect)
            painter.fillRect(corner, color)

            # Label
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            painter.drawText(rect.x() + 4, rect.y() + 14, name)

            # Coords
            painter.setFont(QFont("Arial", 8))
            r = roi['rect']
            painter.drawText(rect.x() + 4, rect.bottom() - 4,
                           f"({r.x()}, {r.y()}, {r.width()}, {r.height()})")

        # Draw zones on minimap
        if 'minimap' in self.rois:
            mm = self.rois['minimap']['rect']
            for zone_name, (x1, y1, x2, y2) in self.zones.items():
                abs_rect = QRect(mm.x() + x1, mm.y() + y1, x2 - x1, y2 - y1)
                disp_rect = self.scaled_rect(abs_rect)
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                painter.drawRect(disp_rect)
                painter.drawText(disp_rect.x() + 2, disp_rect.y() + 10, zone_name[:6])

        # Drawing zone preview
        if self.drawing_zone and self.zone_start:
            painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DashLine))
            preview = QRect(self.zone_start, self.mouse_pos).normalized()
            painter.drawRect(preview)

        # Mouse coords
        real = self.real_pos(self.mouse_pos)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(10, 20, f"Pos: ({real.x()}, {real.y()})")

        painter.end()

    def get_roi_at(self, pos):
        """Find which ROI is under the mouse"""
        real = self.real_pos(pos)
        for name, roi in self.rois.items():
            if roi['rect'].contains(real):
                return name
        return None

    def in_resize_corner(self, pos, name):
        """Check if pos is in resize corner of named ROI"""
        rect = self.scaled_rect(self.rois[name]['rect'])
        return self.corner_rect(rect).contains(pos)

    def mousePressEvent(self, event):
        pos = event.pos()

        if self.drawing_zone:
            # Check if clicking inside minimap
            if 'minimap' in self.rois:
                mm_rect = self.scaled_rect(self.rois['minimap']['rect'])
                if mm_rect.contains(pos):
                    self.zone_start = pos
            return

        # Check ROIs (reverse order for z-order)
        for name in reversed(list(self.rois.keys())):
            if self.in_resize_corner(pos, name):
                self.selected_roi = name
                self.resizing = True
                return

            rect = self.scaled_rect(self.rois[name]['rect'])
            if rect.contains(pos):
                self.selected_roi = name
                self.dragging = True
                self.drag_offset = pos - rect.topLeft()
                return

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()

        if self.dragging and self.selected_roi:
            new_pos = self.real_pos(event.pos() - self.drag_offset)
            self.rois[self.selected_roi]['rect'].moveTopLeft(new_pos)
            self.update()
            self.main_window.update_roi_list()

        elif self.resizing and self.selected_roi:
            real = self.real_pos(event.pos())
            rect = self.rois[self.selected_roi]['rect']
            new_w = max(50, real.x() - rect.x())
            new_h = max(30, real.y() - rect.y())
            rect.setWidth(new_w)
            rect.setHeight(new_h)
            self.update()
            self.main_window.update_roi_list()

        else:
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing_zone and self.zone_start and self.current_zone_name:
            # Save zone (relative to minimap)
            if 'minimap' in self.rois:
                mm = self.rois['minimap']['rect']
                start_real = self.real_pos(self.zone_start)
                end_real = self.real_pos(event.pos())

                x1 = start_real.x() - mm.x()
                y1 = start_real.y() - mm.y()
                x2 = end_real.x() - mm.x()
                y2 = end_real.y() - mm.y()

                self.zones[self.current_zone_name] = (
                    min(x1, x2), min(y1, y2),
                    max(x1, x2), max(y1, y2)
                )
                self.main_window.update_zone_list()

            self.zone_start = None
            self.drawing_zone = False
            self.current_zone_name = None
            self.main_window.zone_btn.setText("Draw Zone")

        self.dragging = False
        self.resizing = False
        self.update()


class CalibrationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("gclear - ROI Calibration")
        self.setGeometry(100, 100, 1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left: image with ROIs
        scroll = QScrollArea()
        self.canvas = CalibrationWidget(main_window=self, parent=scroll)
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        layout.addWidget(scroll, stretch=3)

        # Right: controls
        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)

        # Load image
        load_btn = QPushButton("Load Screenshot")
        load_btn.clicked.connect(self.load_image)
        ctrl_layout.addWidget(load_btn)

        # ROI list
        ctrl_layout.addWidget(QLabel("ROIs (drag to move, corner to resize):"))
        self.roi_list = QListWidget()
        self.roi_list.setStyleSheet("font-family: monospace;")
        ctrl_layout.addWidget(self.roi_list)
        self.update_roi_list()

        # Zone controls
        ctrl_layout.addWidget(QLabel("Minimap Zones:"))

        zone_btns = QHBoxLayout()
        for name in ["blue", "gromp", "wolves", "rapt", "red", "krugs", "spawn"]:
            btn = QPushButton(name)
            btn.setFixedWidth(50)
            full_name = {"blue": "blue_buff", "rapt": "raptors", "red": "red_buff"}.get(name, name)
            btn.clicked.connect(lambda checked, n=full_name: self.start_zone(n))
            zone_btns.addWidget(btn)
        ctrl_layout.addLayout(zone_btns)

        self.zone_btn = QPushButton("Draw Zone")
        self.zone_btn.setEnabled(False)
        ctrl_layout.addWidget(self.zone_btn)

        self.zone_list = QListWidget()
        self.zone_list.setStyleSheet("font-family: monospace;")
        self.zone_list.setMaximumHeight(120)
        ctrl_layout.addWidget(self.zone_list)

        # Export
        export_btn = QPushButton("Export Config")
        export_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        export_btn.clicked.connect(self.export_config)
        ctrl_layout.addWidget(export_btn)

        self.output = QLabel("Click Export to generate config")
        self.output.setStyleSheet("font-family: monospace; font-size: 9px; background: #1a1a1a; color: #0f0; padding: 8px;")
        self.output.setWordWrap(True)
        self.output.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.output.setMinimumHeight(200)
        ctrl_layout.addWidget(self.output)

        ctrl_layout.addStretch()
        layout.addWidget(controls, stretch=1)

        # Try to load a frame around the middle of the dataset
        frames = sorted(Path("data/raw/hecarim_clear").glob("*.png"))
        if frames:
            idx = min(30, len(frames) - 1)
            self.canvas.load_image(str(frames[idx]))

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Screenshot", "data/raw", "Images (*.png *.jpg)")
        if path:
            self.canvas.load_image(path)

    def update_roi_list(self):
        self.roi_list.clear()
        for name, roi in self.canvas.rois.items():
            r = roi['rect']
            self.roi_list.addItem(f"{name}: ({r.x()}, {r.y()}, {r.width()}, {r.height()})")

    def update_zone_list(self):
        self.zone_list.clear()
        for name, coords in self.canvas.zones.items():
            self.zone_list.addItem(f"{name}: {coords}")

    def start_zone(self, name):
        self.canvas.current_zone_name = name
        self.canvas.drawing_zone = True
        self.zone_btn.setText(f"Drawing: {name} (click minimap)")

    def export_config(self):
        lines = ["# Updated ROIs"]
        lines.append("ROIS = {")
        for name, roi in self.canvas.rois.items():
            r = roi['rect']
            lines.append(f'    "{name}": ({r.x()}, {r.y()}, {r.width()}, {r.height()}),')
        lines.append("}")
        lines.append("")
        lines.append("# Minimap zones (relative to minimap ROI)")
        lines.append("MINIMAP_ZONES = {")
        for name, coords in self.canvas.zones.items():
            lines.append(f'    "{name}": {coords},')
        lines.append("}")

        output = "\n".join(lines)
        self.output.setText(output)
        print("\n" + output + "\n")


def main():
    app = QApplication(sys.argv)
    window = CalibrationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
