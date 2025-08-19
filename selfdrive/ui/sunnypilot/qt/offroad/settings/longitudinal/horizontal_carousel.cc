#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/horizontal_carousel.h"
#include <QPainter>
#include <QMouseEvent>
#include <QDebug>
#include <QDateTime>
#include <QtMath>
#include <QEasingCurve>
#include <QFontMetrics>
#include <algorithm>
#include <cmath>

// ============================================================================
// CarouselItem Implementation
// ============================================================================

CarouselItem::CarouselItem(const QString &text, int index, QWidget *parent) 
  : QWidget(parent), index(index), textWidth(0) {
  
  label = new QLabel(text, this);
  label->setAlignment(Qt::AlignCenter);
  label->setWordWrap(false);
  label->setStyleSheet("background-color: transparent; border: none;");
  
  opacityEffect = new QGraphicsOpacityEffect(this);
  label->setGraphicsEffect(opacityEffect);
  
  // Start with minimum scale
  updateScale(1.0f);
  updateTextWidth();
}

void CarouselItem::updateScale(float centerDistance) {
  // centerDistance is 0 when item is at center, increases as it moves away
  // Tighter event horizon for more pronounced lensing effect (was 200.0f)
  float normalizedDistance = qBound(0.0f, qAbs(centerDistance) / 100.0f, 1.0f);
  
  // More dramatic scaling with exponential curve for lensing effect
  float rawScale = 1.0f - normalizedDistance;
  float scale = MIN_SCALE + (rawScale * rawScale * (MAX_SCALE - MIN_SCALE));
  
  // Calculate opacity (more opaque when closer to center)
  float opacity = MAX_OPACITY - (normalizedDistance * (MAX_OPACITY - MIN_OPACITY));
  
  // Apply smooth easing curve for more natural feel
  float easedScale = QEasingCurve(QEasingCurve::OutQuad).valueForProgress(scale);
  float easedOpacity = QEasingCurve(QEasingCurve::OutQuad).valueForProgress(opacity);
  
  // Calculate font size based on scale - larger base for more dramatic effect
  int baseFontSize = 48;
  int fontSize = static_cast<int>(baseFontSize * easedScale);
  
  // Update label style with normal weight for non-selected (400 vs 600)
  QString fontWeight = (normalizedDistance < 0.1f) ? "600" : "400";
  QString color = (normalizedDistance < 0.1f) ? "#FFFFFF" : "#E4E4E4";
  
  label->setStyleSheet(QString(R"(
    QLabel {
      font-size: %1px;
      font-weight: %2;
      color: %3;
      background-color: transparent;
      border: none;
    }
  )").arg(fontSize).arg(fontWeight).arg(color));
  
  opacityEffect->setOpacity(easedOpacity);
  
  // Update widget size to match content
  label->adjustSize();
  resize(label->size());
}

void CarouselItem::setText(const QString &text) {
  label->setText(text);
  label->adjustSize();
  resize(label->size());
  updateTextWidth();
}

void CarouselItem::updateTextWidth() {
  // Calculate actual text width at maximum scale
  QFont font = label->font();
  font.setPixelSize(48);  // Base font size at max scale
  font.setWeight(QFont::DemiBold);  // Selected weight for max width
  QFontMetrics fm(font);
  textWidth = fm.horizontalAdvance(label->text());
}

// ============================================================================
// HorizontalCarousel Implementation
// ============================================================================

HorizontalCarousel::HorizontalCarousel(const QStringList &items, int defaultIndex, QWidget *parent)
  : QWidget(parent), itemTexts(items), currentItemIndex(defaultIndex) {
  
  setFixedHeight(120);
  setAttribute(Qt::WA_TranslucentBackground);
  
  // Create carousel items
  for (int i = 0; i < items.size(); ++i) {
    auto item = std::make_unique<CarouselItem>(items[i], i, this);
    carouselItems.push_back(std::move(item));
  }
  
  // Calculate dynamic positions based on text widths
  calculateDynamicPositions();
  
  // Setup animation
  snapAnimation = new QPropertyAnimation(this, "scrollPosition", this);
  snapAnimation->setEasingCurve(QEasingCurve::OutCubic);
  snapAnimation->setDuration(SNAP_ANIMATION_DURATION);
  connect(snapAnimation, &QPropertyAnimation::finished, this, &HorizontalCarousel::onAnimationFinished);
  
  // Setup physics timer
  physicsTimer = new QTimer(this);
  physicsTimer->setInterval(16); // ~60 FPS
  connect(physicsTimer, &QTimer::timeout, this, &HorizontalCarousel::updatePhysics);
  
  // Set initial position
  currentScrollPosition = calculateSnapPosition(defaultIndex);
  targetScrollPosition = currentScrollPosition;
  
  // Initial layout
  layoutItems();
}

QString HorizontalCarousel::currentText() const {
  if (currentItemIndex >= 0 && currentItemIndex < itemTexts.size()) {
    return itemTexts[currentItemIndex];
  }
  return QString();
}

void HorizontalCarousel::setCurrentIndex(int index, bool animated) {
  if (index < 0 || index >= carouselItems.size()) return;
  if (index == currentItemIndex && !isDragging) return;
  
  int oldIndex = currentItemIndex;
  currentItemIndex = index;
  targetScrollPosition = calculateSnapPosition(index);
  
  if (animated && !isDragging) {
    snapAnimation->stop();
    snapAnimation->setStartValue(currentScrollPosition);
    snapAnimation->setEndValue(targetScrollPosition);
    snapAnimation->start();
  } else {
    currentScrollPosition = targetScrollPosition;
    layoutItems();
  }
  
  if (oldIndex != currentItemIndex) {
    emit currentIndexChanged(currentItemIndex);
    emit currentTextChanged(currentText());
  }
}

void HorizontalCarousel::resizeEvent(QResizeEvent *event) {
  QWidget::resizeEvent(event);
  layoutItems();
}

void HorizontalCarousel::paintEvent(QPaintEvent *event) {
  QPainter painter(this);
  painter.setRenderHint(QPainter::Antialiasing);
  
  // Draw subtle gradient overlay at edges for depth
  int fadeWidth = 100;
  QLinearGradient leftGradient(0, 0, fadeWidth, 0);
  leftGradient.setColorAt(0.0, QColor(32, 32, 32, 180));
  leftGradient.setColorAt(1.0, QColor(32, 32, 32, 0));
  painter.fillRect(0, 0, fadeWidth, height(), leftGradient);
  
  QLinearGradient rightGradient(width() - fadeWidth, 0, width(), 0);
  rightGradient.setColorAt(0.0, QColor(32, 32, 32, 0));
  rightGradient.setColorAt(1.0, QColor(32, 32, 32, 180));
  painter.fillRect(width() - fadeWidth, 0, fadeWidth, height(), rightGradient);
  
  // Draw selection indicator (subtle line below selected item)
  int centerX = width() / 2;
  int indicatorY = height() - 15;
  int indicatorWidth = 60;
  int indicatorHeight = 3;
  
  // Gradient for selection indicator
  QLinearGradient indicatorGradient(centerX - indicatorWidth/2, 0, centerX + indicatorWidth/2, 0);
  indicatorGradient.setColorAt(0.0, QColor(255, 193, 7, 0));
  indicatorGradient.setColorAt(0.2, QColor(255, 193, 7, 200));
  indicatorGradient.setColorAt(0.5, QColor(255, 193, 7, 255));
  indicatorGradient.setColorAt(0.8, QColor(255, 193, 7, 200));
  indicatorGradient.setColorAt(1.0, QColor(255, 193, 7, 0));
  
  painter.fillRect(centerX - indicatorWidth/2, indicatorY, indicatorWidth, indicatorHeight, indicatorGradient);
}

void HorizontalCarousel::mousePressEvent(QMouseEvent *event) {
  if (event->button() != Qt::LeftButton) return;
  
  isDragging = true;
  touchStartX = event->x();
  lastTouchX = event->x();
  touchStartTime = QDateTime::currentMSecsSinceEpoch();
  lastTouchTime = touchStartTime;
  
  // Clear touch history
  touchHistory.clear();
  touchHistory.append({event->x(), touchStartTime});
  
  // Stop any ongoing animation
  snapAnimation->stop();
  physicsTimer->stop();
  scrollVelocity = 0.0f;
}

void HorizontalCarousel::mouseMoveEvent(QMouseEvent *event) {
  if (!isDragging) return;
  
  qint64 currentTime = QDateTime::currentMSecsSinceEpoch();
  int deltaX = event->x() - lastTouchX;
  
  // Calculate proposed new position
  float proposedPosition = currentScrollPosition + deltaX;
  
  // Apply elastic boundaries with resistance
  float minScroll = 0.0f;
  float maxScroll = itemPositions.empty() ? 0.0f : itemPositions.back();
  
  if (proposedPosition < minScroll) {
    // Elastic resistance when pulling past first item
    float overscroll = minScroll - proposedPosition;
    float resistance = 1.0f / (1.0f + overscroll * 0.001f);  // Exponential resistance
    currentScrollPosition = minScroll - (overscroll * resistance * 0.3f);  // Max 30% of overscroll
  } else if (proposedPosition > maxScroll) {
    // Elastic resistance when pulling past last item
    float overscroll = proposedPosition - maxScroll;
    float resistance = 1.0f / (1.0f + overscroll * 0.001f);
    currentScrollPosition = maxScroll + (overscroll * resistance * 0.3f);
  } else {
    // Normal scrolling within bounds
    currentScrollPosition = proposedPosition;
  }
  
  // Add to touch history for velocity calculation
  touchHistory.append({event->x(), currentTime});
  while (touchHistory.size() > TOUCH_HISTORY_SIZE) {
    touchHistory.removeFirst();
  }
  
  lastTouchX = event->x();
  lastTouchTime = currentTime;
  
  layoutItems();
}

void HorizontalCarousel::mouseReleaseEvent(QMouseEvent *event) {
  if (!isDragging) return;
  
  isDragging = false;
  
  // Check if we're outside boundaries (elastic snap-back needed)
  float minScroll = 0.0f;
  float maxScroll = itemPositions.empty() ? 0.0f : itemPositions.back();
  
  if (currentScrollPosition < minScroll || currentScrollPosition > maxScroll) {
    // Elastic snap-back to boundary
    if (currentScrollPosition < minScroll) {
      setCurrentIndex(0, true);
    } else {
      setCurrentIndex(carouselItems.size() - 1, true);
    }
    return;
  }
  
  // Calculate velocity from touch history
  scrollVelocity = 0.0f;
  if (touchHistory.size() >= 2) {
    auto recent = touchHistory.last();
    auto previous = touchHistory[touchHistory.size() - 2];
    
    qint64 timeDelta = recent.second - previous.second;
    if (timeDelta > 0) {
      float pixelDelta = recent.first - previous.first;
      scrollVelocity = (pixelDelta / timeDelta) * 1000.0f * VELOCITY_MULTIPLIER;
    }
  }
  
  // Determine target item based on velocity and position
  int targetIndex = findNearestItem();
  
  // If we have significant velocity, consider snapping to next/previous item
  if (qAbs(scrollVelocity) > 200.0f) {  // Velocity threshold for skipping to next
    if (scrollVelocity < 0) {
      // Scrolling left (increasing index)
      targetIndex = qMin(targetIndex + 1, static_cast<int>(carouselItems.size() - 1));
    } else {
      // Scrolling right (decreasing index)
      targetIndex = qMax(targetIndex - 1, 0);
    }
  }
  
  // Always snap to an item (magnetic behavior)
  setCurrentIndex(targetIndex, true);
}

void HorizontalCarousel::updatePhysics() {
  if (isDragging) return;
  
  // Apply velocity to position
  currentScrollPosition += scrollVelocity * 0.016f; // 16ms frame time
  
  // Apply boundary constraints with bounce-back
  float minScroll = 0.0f;
  float maxScroll = itemPositions.empty() ? 0.0f : itemPositions.back();
  
  if (currentScrollPosition < minScroll) {
    currentScrollPosition = minScroll;
    scrollVelocity = 0.0f;  // Stop at boundary
    physicsTimer->stop();
    setCurrentIndex(0, true);
    return;
  } else if (currentScrollPosition > maxScroll) {
    currentScrollPosition = maxScroll;
    scrollVelocity = 0.0f;  // Stop at boundary
    physicsTimer->stop();
    setCurrentIndex(carouselItems.size() - 1, true);
    return;
  }
  
  // Apply deceleration (friction)
  scrollVelocity *= DECELERATION_RATE;
  
  // Always snap when velocity gets low (stronger magnetism)
  if (qAbs(scrollVelocity) < 50.0f) {  // Increased threshold for earlier snapping
    physicsTimer->stop();
    snapToNearestItem();
  } else {
    layoutItems();
  }
}

void HorizontalCarousel::onAnimationFinished() {
  // Ensure we're exactly at the target position
  currentScrollPosition = targetScrollPosition;
  layoutItems();
}

void HorizontalCarousel::layoutItems() {
  int centerX = width() / 2;
  int centerY = height() / 2;
  
  for (size_t i = 0; i < carouselItems.size(); ++i) {
    auto& item = carouselItems[i];
    
    // Use dynamic position from itemPositions
    float itemCenterX = centerX + itemPositions[i] - currentScrollPosition;
    float distanceFromCenter = itemCenterX - centerX;
    
    // Update item scale and opacity based on distance from center
    item->updateScale(distanceFromCenter);
    
    // Position item
    int itemX = static_cast<int>(itemCenterX - item->width() / 2);
    int itemY = centerY - item->height() / 2;
    item->move(itemX, itemY);
    
    // Show/hide based on visibility
    bool isVisible = (itemX + item->width() > -100) && (itemX < width() + 100);
    item->setVisible(isVisible);
  }
  
  update(); // Trigger repaint for selection indicator
}

void HorizontalCarousel::updateItemScales() {
  layoutItems();
}

void HorizontalCarousel::snapToNearestItem() {
  int nearestIndex = findNearestItem();
  setCurrentIndex(nearestIndex, true);
}

float HorizontalCarousel::calculateSnapPosition(int index) {
  // Calculate the scroll position that centers the item at the given index
  if (index >= 0 && index < static_cast<int>(itemPositions.size())) {
    return itemPositions[index];
  }
  return 0.0f;
}

int HorizontalCarousel::findNearestItem() {
  // Find which item is closest to the center
  int nearestIndex = 0;
  float minDistance = std::numeric_limits<float>::max();
  
  for (size_t i = 0; i < itemPositions.size(); ++i) {
    float distance = qAbs(itemPositions[i] - currentScrollPosition);
    
    if (distance < minDistance) {
      minDistance = distance;
      nearestIndex = static_cast<int>(i);
    }
  }
  
  return nearestIndex;
}

void HorizontalCarousel::startInertialScroll() {
  physicsTimer->start();
}

void HorizontalCarousel::setScrollPosition(float pos) {
  currentScrollPosition = pos;
  layoutItems();
}

void HorizontalCarousel::calculateDynamicPositions() {
  itemPositions.clear();
  
  if (carouselItems.empty()) return;
  
  float currentPos = 0.0f;
  
  for (size_t i = 0; i < carouselItems.size(); ++i) {
    // Center position for this item
    itemPositions.push_back(currentPos);
    
    if (i < carouselItems.size() - 1) {
      // Calculate position for next item based on this item's width + uniform spacing
      int thisItemHalfWidth = carouselItems[i]->getTextWidth() / 2;
      int nextItemHalfWidth = carouselItems[i + 1]->getTextWidth() / 2;
      
      // Ensure minimum width for very short text
      thisItemHalfWidth = std::max(thisItemHalfWidth, MIN_ITEM_WIDTH / 2);
      nextItemHalfWidth = std::max(nextItemHalfWidth, MIN_ITEM_WIDTH / 2);
      
      // Distance to next item center = half of this item + spacing + half of next item
      currentPos += thisItemHalfWidth + UNIFORM_EDGE_SPACING + nextItemHalfWidth;
    }
  }
}

// ============================================================================
// RTIThreatFilterCarousel Implementation
// ============================================================================

RTIThreatFilterCarousel::RTIThreatFilterCarousel(QWidget *parent)
  : HorizontalCarousel({
      tr("All"),
      tr("Police"),
      tr("Cameras"),
      tr("Hazards"),
      tr("Custom")
    }, 0, parent) {
  
  // Load current setting
  int filterVal = QString::fromStdString(params.get("RTIThreatFilter")).toInt();
  setCurrentIndex(filterVal, false);
  
  // Connect signal to save changes
  connect(this, &HorizontalCarousel::currentIndexChanged, this, &RTIThreatFilterCarousel::onSelectionChanged);
}

void RTIThreatFilterCarousel::onSelectionChanged(int index) {
  params.put("RTIThreatFilter", std::to_string(index));
}

// ============================================================================
// RTISpeedModeCarousel Implementation
// ============================================================================

RTISpeedModeCarousel::RTISpeedModeCarousel(QWidget *parent)
  : HorizontalCarousel({
      tr("Posted"),
      tr("Custom")
    }, 0, parent) {
  
  // Load current setting
  QString speedMode = QString::fromStdString(params.get("RTISpeedReductionMode"));
  setCurrentIndex(speedMode == "custom" ? 1 : 0, false);
  
  // Connect signal to save changes
  connect(this, &HorizontalCarousel::currentIndexChanged, this, &RTISpeedModeCarousel::onSelectionChanged);
}

void RTISpeedModeCarousel::onSelectionChanged(int index) {
  params.put("RTISpeedReductionMode", index == 0 ? "posted" : "custom");
}