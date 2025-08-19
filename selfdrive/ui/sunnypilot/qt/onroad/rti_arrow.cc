/**
 * RTI Arrow Implementation
 */

#include "selfdrive/ui/sunnypilot/qt/onroad/rti_arrow.h"
#include <QPainterPath>
#include <QTransform>

RTIArrow::RTIArrow(QWidget *parent) : QWidget(parent) {
  setFixedSize(kArrowSize, kArrowSize);
  setAttribute(Qt::WA_TranslucentBackground);
  createArrowPixmap();
}

void RTIArrow::createArrowPixmap() {
  // Create arrow pointing up (0° = ahead)
  arrow_pixmap = QPixmap(kArrowSize, kArrowSize);
  arrow_pixmap.fill(Qt::transparent);
  
  QPainter painter(&arrow_pixmap);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.setRenderHint(QPainter::SmoothPixmapTransform);
  
  // Create arrow path (pointing up)
  QPainterPath arrow;
  int center = kArrowSize / 2;
  int arrow_length = kArrowSize * 0.8;
  int arrow_width = kArrowSize * 0.5;
  
  // Arrow tip (top)
  arrow.moveTo(center, center - arrow_length/2);
  
  // Right side of arrowhead
  arrow.lineTo(center + arrow_width/3, center - arrow_length/6);
  
  // Right side of shaft
  arrow.lineTo(center + arrow_width/6, center - arrow_length/6);
  arrow.lineTo(center + arrow_width/6, center + arrow_length/3);
  
  // Bottom of arrow
  arrow.lineTo(center - arrow_width/6, center + arrow_length/3);
  
  // Left side of shaft
  arrow.lineTo(center - arrow_width/6, center - arrow_length/6);
  
  // Left side of arrowhead
  arrow.lineTo(center - arrow_width/3, center - arrow_length/6);
  
  // Close path back to tip
  arrow.closeSubpath();
  
  // Fill with white (will be tinted based on threat color)
  painter.fillPath(arrow, Qt::white);
  
  // Add border for better visibility
  painter.setPen(QPen(QColor(0, 0, 0, 100), 2));
  painter.drawPath(arrow);
  
  pixmap_cached = true;
}

void RTIArrow::setRotation(double bearing_deg) {
  rotation_angle = bearing_deg;
  update();  // Trigger repaint
}

void RTIArrow::setThreatType(int type) {
  threat_type = type;
  update();
}

void RTIArrow::setVisible(bool visible) {
  is_visible = visible;
  QWidget::setVisible(visible);
}

QColor RTIArrow::getThreatColor(double distance_m) const {
  // Color based on distance
  if (distance_m < 200) {
    return QColor(255, 0, 0, 255);     // Red - Critical
  } else if (distance_m < 500) {
    return QColor(255, 165, 0, 255);   // Orange - Near
  } else if (distance_m < 1000) {
    return QColor(255, 255, 0, 255);   // Yellow - Normal
  } else {
    return QColor(150, 150, 150, 255); // Gray - Far
  }
}

void RTIArrow::paintEvent(QPaintEvent *event) {
  if (!is_visible || !pixmap_cached) {
    return;
  }
  
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);
  painter.setRenderHint(QPainter::SmoothPixmapTransform);
  
  // Apply rotation around center
  QTransform transform;
  transform.translate(width() / 2.0, height() / 2.0);
  transform.rotate(rotation_angle);
  transform.translate(-width() / 2.0, -height() / 2.0);
  
  painter.setTransform(transform);
  
  // Draw the rotated arrow
  painter.drawPixmap(0, 0, arrow_pixmap);
}