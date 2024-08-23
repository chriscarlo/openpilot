#pragma once

#include <QVBoxLayout>
#include <memory>

#include "selfdrive/ui/qt/onroad/buttons.h"
#include "selfdrive/ui/qt/widgets/cameraview.h"

#include "selfdrive/frogpilot/screenrecorder/screenrecorder.h"

class Compass : public QWidget {
  Q_OBJECT

public:
  explicit Compass(QWidget *parent = 0);
  void updateState(const UIScene &scene);

private:
  void initializeStaticElements();
  void paintEvent(QPaintEvent *event) override;

  int bearingDeg;
  int circleOffset;
  int compassSize;
  int degreeLabelOffset;
  int innerCompass;
  int x;
  int y;

  QPixmap compassInnerImg;
  QPixmap staticElements;
};

class PedalIcons : public QWidget {
  Q_OBJECT

public:
  explicit PedalIcons(QWidget *parent = 0);
  void updateState(const UIScene &scene);

private:
  void paintEvent(QPaintEvent *event) override;

  QPixmap brake_pedal_img;
  QPixmap gas_pedal_img;

  bool accelerating;
  bool brakeLightOn;
  bool decelerating;
  bool dynamicPedals;
  bool standstill;
  bool staticPedals;

  float acceleration;
};

class AnnotatedCameraWidget : public CameraWidget {
  Q_OBJECT

public:
  explicit AnnotatedCameraWidget(VisionStreamType type, QWidget* parent = 0);
  void updateState(int alert_height, const UIState &s);

  MapSettingsButton *map_settings_btn;
  MapSettingsButton *map_settings_btn_bottom;

private:
  void drawText(QPainter &p, int x, int y, const QString &text, int alpha = 255, bool overridePen = false);

  QVBoxLayout *main_layout;
  ExperimentalButton *experimental_btn;
  QPixmap dm_img;
  float speed;
  QString speedUnit;
  float setSpeed;
  float speedLimit;
  bool is_cruise_set = false;
  bool is_metric = false;
  bool dmActive = false;
  bool hideBottomIcons = false;
  bool rightHandDM = false;
  float dm_fade_state = 1.0;
  bool has_us_speed_limit = false;
  bool has_eu_speed_limit = false;
  bool v_ego_cluster_seen = false;
  int status = STATUS_DISENGAGED;
  std::unique_ptr<PubMaster> pm;

  int skip_frame_count = 0;
  bool wide_cam_requested = false;

  // FrogPilot widgets
  void initializeFrogPilotWidgets();
  void paintFrogPilotWidgets(QPainter &painter);
  void updateFrogPilotWidgets(int alert_height, const UIScene &scene);
  void updateSignals();

  void drawLeadInfo(QPainter &p);
  void drawSLCConfirmation(QPainter &p);
  void drawStatusBar(QPainter &p);
  void drawTurnSignals(QPainter &p);

  // FrogPilot variables
  Params paramsMemory{"/dev/shm/params"};

  //CC
  Params params;
  bool ShowDMIcon = params.getBool("ShowDMIcon");

  Compass *compass_img;
  DistanceButton *distance_btn;
  PedalIcons *pedal_icons;
  ScreenRecorder *recorder;

  QHBoxLayout *bottom_layout;

  bool alwaysOnLateralActive;
  bool bigMapOpen;
  bool blindSpotLeft;
  bool blindSpotRight;
  bool compass;
  bool experimentalMode;
  bool hideMapIcon;
  bool hideMaxSpeed;
  bool hideSpeed;
  bool leadInfo;
  bool mapOpen;
  bool onroadDistanceButton;
  bool reverseCruise;
  bool roadNameUI;
  bool showAlwaysOnLateralStatusBar;
  bool showConditionalExperimentalStatusBar;
  bool showSLCOffset;
  bool slcOverridden;
  bool speedLimitChanged;
  bool speedLimitController;
  bool trafficModeActive;
  bool turnSignalAnimation;
  bool turnSignalLeft;
  bool turnSignalRight;
  bool useStockColors;
  bool useSI;
  bool useViennaSLCSign;
  bool vtscControllingCurve;

  double currentAcceleration;

  float accelerationConversion;
  float cruiseAdjustment;
  float distanceConversion;
  float laneDetectionWidth;
  float laneWidthLeft;
  float laneWidthRight;
  float slcSpeedLimitOffset;
  float speedConversion;
  float unconfirmedSpeedLimit;

  int alertHeight;
  int animationFrameIndex;
  int cameraView;
  int conditionalSpeed;
  int conditionalSpeedLead;
  int conditionalStatus;
  int desiredFollow;
  int modelLength;
  int obstacleDistance;
  int obstacleDistanceStock;
  int signalAnimationLength;
  int signalHeight;
  int signalWidth;
  int speedTextWidth;
  int standstillDuration;
  int statusBarHeight;
  int stoppedEquivalence;
  int totalFrames;

  QElapsedTimer standstillTimer;

  QPixmap stopSignImg;

  QString accelerationUnit;
  QString leadDistanceUnit;
  QString leadSpeedUnit;
  QString signalStyle;

  QTimer *animationTimer;

  std::vector<QPixmap> regularImages;
  std::vector<QPixmap> blindspotImages;

  inline QColor blueColor(int alpha = 255) { return QColor(0, 150, 255, alpha); }
  inline QColor greenColor(int alpha = 242) { return QColor(23, 134, 68, alpha); }

protected:
  void paintGL() override;
  void initializeGL() override;
  void showEvent(QShowEvent *event) override;
  void updateFrameMat() override;
  void drawLaneLines(QPainter &painter, const UIState *s);
  void drawLead(QPainter &painter, const cereal::ModelDataV2::LeadDataV3::Reader &lead_data, const QPointF &vd, const float v_ego, const QColor lead_marker_color);
  void drawHud(QPainter &p);
  void drawDriverState(QPainter &painter, const UIState *s);
  void paintEvent(QPaintEvent *event) override;
  inline QColor redColor(int alpha = 255) { return QColor(201, 34, 49, alpha); }
  inline QColor whiteColor(int alpha = 255) { return QColor(255, 255, 255, alpha); }
  inline QColor blackColor(int alpha = 255) { return QColor(0, 0, 0, alpha); }

  double prev_draw_t = 0;
  FirstOrderFilter fps_filter;
};
