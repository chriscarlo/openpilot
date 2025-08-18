#!/usr/bin/env python3
"""
Visual test of RTI arrow icon to verify sizing and appearance
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPainter, QPainterPath, QPixmap, QFont, QColor, QPen, QBrush, QTransform
from PyQt5.QtCore import Qt, QRect, QPoint

class ArrowVisualization(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('RTI Arrow Icon Visualization')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #1a1a1a;")  # Dark background like HUD
        
    def create_arrow_pixmap(self, size=48):
        """Create the arrow pixmap exactly as in C++ implementation"""
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create arrow path (pointing up)
        arrow = QPainterPath()
        center = size // 2
        arrow_length = int(size * 0.8)
        arrow_width = int(size * 0.5)
        
        # Arrow tip (top)
        arrow.moveTo(center, center - arrow_length//2)
        
        # Right side of arrowhead
        arrow.lineTo(center + arrow_width//3, center - arrow_length//6)
        
        # Right side of shaft
        arrow.lineTo(center + arrow_width//6, center - arrow_length//6)
        arrow.lineTo(center + arrow_width//6, center + arrow_length//3)
        
        # Bottom of arrow
        arrow.lineTo(center - arrow_width//6, center + arrow_length//3)
        
        # Left side of shaft
        arrow.lineTo(center - arrow_width//6, center - arrow_length//6)
        
        # Left side of arrowhead
        arrow.lineTo(center - arrow_width//3, center - arrow_length//6)
        
        # Close path back to tip
        arrow.closePath()
        
        # Fill with white (will be tinted when drawn)
        painter.fillPath(arrow, Qt.white)
        
        # Add border for better visibility
        painter.setPen(QPen(QColor(0, 0, 0, 100), 2))
        painter.drawPath(arrow)
        
        painter.end()
        return pixmap
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw title
        painter.setPen(QColor(255, 255, 255))
        title_font = QFont("Arial", 20, QFont.Bold)
        painter.setFont(title_font)
        painter.drawText(20, 40, "RTI Arrow Icon - Size Comparison")
        
        # Font sizes from actual implementation
        threat_font_size = 42  # From InterFont(42, QFont::Normal)
        
        # Threat colors by distance
        colors = [
            ("Critical (<200m)", QColor(255, 0, 0, 255)),      # Red
            ("Near (200-500m)", QColor(255, 165, 0, 255)),     # Orange  
            ("Normal (500-1000m)", QColor(255, 255, 0, 255)),  # Yellow
            ("Far (>1000m)", QColor(150, 150, 150, 255))       # Gray
        ]
        
        y_offset = 100
        
        for i, (label, color) in enumerate(colors):
            x_base = 50
            
            # Draw distance label
            painter.setPen(QColor(200, 200, 200))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(x_base, y_offset - 10, label)
            
            # Create and draw arrow at different rotations
            rotations = [0, 45, 90, 135, 180, 225, 270, 315]
            directions = ["↑ Ahead", "↗ Ahead-Right", "→ Right", "↘ Behind-Right", 
                         "↓ Behind", "↙ Behind-Left", "← Left", "↖ Ahead-Left"]
            
            for j, (rot, dir_label) in enumerate(zip(rotations, directions)):
                x = x_base + j * 90
                
                # Draw arrow with rotation
                arrow_pixmap = self.create_arrow_pixmap(48)
                
                # Tint the arrow
                tinted = QPixmap(arrow_pixmap.size())
                tinted.fill(Qt.transparent)
                tint_painter = QPainter(tinted)
                tint_painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                tint_painter.drawPixmap(0, 0, arrow_pixmap)
                tint_painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                tint_painter.fillRect(tinted.rect(), color)
                tint_painter.end()
                
                # Apply rotation
                transform = QTransform()
                transform.rotate(rot)
                rotated = tinted.transformed(transform, Qt.SmoothTransformation)
                
                # Draw the rotated arrow
                painter.drawPixmap(x, y_offset, rotated)
                
                # Draw direction label
                painter.setPen(QColor(150, 150, 150))
                painter.setFont(QFont("Arial", 9))
                painter.drawText(x - 10, y_offset + 65, dir_label)
            
            y_offset += 100
        
        # Draw example RTI widget mockup
        y_offset += 50
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(title_font)
        painter.drawText(20, y_offset, "RTI Widget Mockup (Actual Size)")
        
        # Draw RTI widget background
        widget_rect = QRect(50, y_offset + 20, 525, 365)
        painter.setPen(QPen(QColor(255, 255, 255, 75), 6))
        painter.setBrush(QColor(0, 0, 0, 115))
        painter.drawRoundedRect(widget_rect, 32, 32)
        
        # Draw threat with arrow
        threat_color = QColor(255, 165, 0, 255)  # Orange for example
        
        # Arrow position (as in actual code)
        arrow_rect = QRect(widget_rect.x() + 180, widget_rect.y() + 125, 48, 48)
        arrow_pixmap = self.create_arrow_pixmap(48)
        
        # Tint and rotate arrow (45 degrees for example)
        tinted = QPixmap(arrow_pixmap.size())
        tinted.fill(Qt.transparent)
        tint_painter = QPainter(tinted)
        tint_painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        tint_painter.drawPixmap(0, 0, arrow_pixmap)
        tint_painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        tint_painter.fillRect(tinted.rect(), threat_color)
        tint_painter.end()
        
        transform = QTransform()
        transform.rotate(45)  # Example: threat ahead-right
        rotated = tinted.transformed(transform, Qt.SmoothTransformation)
        
        painter.drawPixmap(arrow_rect.x(), arrow_rect.y(), rotated)
        
        # Draw threat text (shifted right as in actual code)
        painter.setFont(QFont("Arial", threat_font_size))
        painter.setPen(threat_color)
        text_rect = QRect(widget_rect.x() + 50, widget_rect.y() + 132, 
                         widget_rect.width() - 100, 50)
        painter.drawText(text_rect, Qt.AlignTop | Qt.AlignHCenter, "POLICE")
        
        # Draw distance text
        painter.setFont(QFont("Arial", 45, QFont.Bold))
        distance_rect = QRect(widget_rect.x(), widget_rect.y() + 186, 
                            widget_rect.width(), 50)
        painter.drawText(distance_rect, Qt.AlignTop | Qt.AlignHCenter, "0.5mi")
        
        # Draw size comparison note
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Arial", 11))
        note = "Arrow: 48x48px | Threat Text: 42px | Distance Text: 45px"
        painter.drawText(50, widget_rect.bottom() + 30, note)

def main():
    app = QApplication(sys.argv)
    viz = ArrowVisualization()
    viz.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()