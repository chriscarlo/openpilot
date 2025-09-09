// Minimal Bluetooth panel scaffolding for Offroad Network → Bluetooth
#pragma once

#include <QWidget>
#include <QStackedLayout>
#include <QLabel>
#include <QTimer>
#include <QProcess>
#include <QFrame>
#include <QPushButton>
#include <QVector>
#include <QString>
#include <functional>



#ifdef SUNNYPILOT
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#else
#include "selfdrive/ui/qt/widgets/controls.h"
#endif

struct BtDevice {
  QString addr;
  QString name;
  int rssi = 0;
  bool paired = false;
  bool trusted = false;
  bool connected = false;
};

class DeviceItem : public QWidget {
  Q_OBJECT
public:
  explicit DeviceItem(QWidget *parent = nullptr);
  void setDevice(const BtDevice &d);

signals:
  void pairRequested(const QString &addr);
  void connectRequested(const QString &addr);
  void disconnectRequested(const QString &addr);
  void trustRequested(const QString &addr, bool trust);
  void removeRequested(const QString &addr);

private:
  BtDevice dev;
  QLabel *title = nullptr;
  QLabel *subtitle = nullptr;
  QPushButton *pairBtn = nullptr;
  QPushButton *connBtn = nullptr;
  QPushButton *trustBtn = nullptr;
  QPushButton *removeBtn = nullptr;
  void updateButtons();
};

class BluetoothPanel : public QWidget {
  Q_OBJECT

public:
  explicit BluetoothPanel(QWidget *parent = nullptr);

signals:
  void backPress();

private:
  Params params;
  LabelControl *status_label = nullptr;
  QTimer *poll_timer = nullptr;
  QPushButton *scanBtn = nullptr;
  ListWidget *device_list_widget = nullptr;
  QVector<BtDevice> devices;
  QFrame *adapterCard = nullptr;
  QFrame *devicesCard = nullptr;
  QLabel *scanStatus = nullptr;
  QLabel *msgLabel = nullptr;
  QProcess *scanProc = nullptr;
  QTimer *scanKillTimer = nullptr;
  bool statusPending = false;

  void updateStatus();
  void startScanOnce();
  void onScanFinished(int exitCode, QProcess::ExitStatus status);
  void refreshDevicesFromJson(const QString &json);
  void rebuildDeviceList();
  int runHelper(const QString &args, int timeout_ms, QString *stdout_str = nullptr);
  void runHelperAsync(const QString &args, int timeout_ms, std::function<void(int, const QString &)> cb);
  void runShellAsync(const QString &cmd, int timeout_ms, std::function<void(int, const QString &)> cb);
  void showMessage(const QString &text, bool is_error = false);
};
