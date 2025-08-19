#pragma once

#include <QWidget>
#include <QStringList>
#include <QTimer>
#include <QPropertyAnimation>
#include <QGraphicsOpacityEffect>
#include <QLabel>
#include <vector>
#include <memory>
#include "common/params.h"

class CarouselItem : public QWidget {
  Q_OBJECT
  
public:
  explicit CarouselItem(const QString &text, int index, QWidget *parent = nullptr);
  void updateScale(float centerDistance);
  void setText(const QString &text);
  QString text() const { return label->text(); }
  int getIndex() const { return index; }
  int getTextWidth() const { return textWidth; }  // Actual text pixel width
  void updateTextWidth();  // Recalculate text width with current font
  
private:
  QLabel *label;
  int index;
  int textWidth;  // Cached text width for layout calculations
  QGraphicsOpacityEffect *opacityEffect;
  
  static constexpr float MIN_SCALE = 0.5f;  // Smaller minimum for more dramatic scaling
  static constexpr float MAX_SCALE = 1.0f;
  static constexpr float MIN_OPACITY = 0.3f;  // Lower opacity for better contrast
  static constexpr float MAX_OPACITY = 1.0f;
};

class HorizontalCarousel : public QWidget {
  Q_OBJECT
  Q_PROPERTY(float scrollPosition READ scrollPosition WRITE setScrollPosition)
  
public:
  explicit HorizontalCarousel(const QStringList &items, int defaultIndex = 0, QWidget *parent = nullptr);
  
  int currentIndex() const { return currentItemIndex; }
  QString currentText() const;
  void setCurrentIndex(int index, bool animated = true);
  
signals:
  void currentIndexChanged(int index);
  void currentTextChanged(const QString &text);
  
protected:
  void resizeEvent(QResizeEvent *event) override;
  void paintEvent(QPaintEvent *event) override;
  
  // Touch handling
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;
  
private slots:
  void updatePhysics();
  void onAnimationFinished();
  
private:
  void layoutItems();
  void updateItemScales();
  void snapToNearestItem();
  float calculateSnapPosition(int index);
  int findNearestItem();
  void startInertialScroll();
  void calculateDynamicPositions();  // Calculate positions based on text widths
  
  float scrollPosition() const { return currentScrollPosition; }
  void setScrollPosition(float pos);
  
  // Items
  std::vector<std::unique_ptr<CarouselItem>> carouselItems;
  std::vector<float> itemPositions;  // Dynamic positions based on text width
  QStringList itemTexts;
  int currentItemIndex = 0;
  
  // Scrolling physics
  float currentScrollPosition = 0.0f;
  float targetScrollPosition = 0.0f;
  float scrollVelocity = 0.0f;
  
  // Touch tracking
  bool isDragging = false;
  int lastTouchX = 0;
  int touchStartX = 0;
  qint64 lastTouchTime = 0;
  qint64 touchStartTime = 0;
  QList<QPair<int, qint64>> touchHistory;
  
  // Animation
  QPropertyAnimation *snapAnimation;
  QTimer *physicsTimer;
  
  // Layout constants - dynamic spacing for uniform edge distances
  static constexpr int UNIFORM_EDGE_SPACING = 80;  // Uniform spacing between text edges
  static constexpr int MIN_ITEM_WIDTH = 100;  // Minimum width for very short text
  static constexpr int VISIBLE_ITEMS = 5;
  
  // Physics constants (iOS-like with stronger magnetism)
  static constexpr float DECELERATION_RATE = 0.88f;  // Stronger friction for quicker stops
  static constexpr float SNAP_VELOCITY_THRESHOLD = 50.0f;  // Snap earlier for stronger magnetism
  static constexpr float SNAP_DISTANCE_THRESHOLD = 30.0f;
  static constexpr int SNAP_ANIMATION_DURATION = 200;  // Snappier animation
  static constexpr float VELOCITY_MULTIPLIER = 2.0f;  // Slightly reduced for better control
  static constexpr int TOUCH_HISTORY_SIZE = 5;
  static constexpr float ELASTIC_RESISTANCE = 0.3f;  // Max elastic overscroll (30%)
};

// Specialized carousels for RTI settings
class RTIThreatFilterCarousel : public HorizontalCarousel {
  Q_OBJECT
  
public:
  explicit RTIThreatFilterCarousel(QWidget *parent = nullptr);
  
private slots:
  void onSelectionChanged(int index);
  
private:
  Params params;
};

class RTISpeedModeCarousel : public HorizontalCarousel {
  Q_OBJECT
  
public:
  explicit RTISpeedModeCarousel(QWidget *parent = nullptr);
  
private slots:
  void onSelectionChanged(int index);
  
private:
  Params params;
};