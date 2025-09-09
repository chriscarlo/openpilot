#include "selfdrive/ui/qt/network/bluetooth.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QProcess>
#include <QPushButton>
#include <QHBoxLayout>

#include <algorithm>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include "selfdrive/ui/qt/util.h"

// DeviceItem implementation
DeviceItem::DeviceItem(QWidget *parent) : QWidget(parent) {
  QHBoxLayout *hl = new QHBoxLayout(this);
  hl->setContentsMargins(0, 0, 0, 0);
  hl->setSpacing(20);
  QVBoxLayout *textCol = new QVBoxLayout();
  textCol->setContentsMargins(0, 0, 0, 0);
  textCol->setSpacing(6);
  title = new QLabel("Device");
  title->setStyleSheet("font-size: 36px; font-weight: 500; color: #FFFFFF;");
  subtitle = new QLabel("addr  •  RSSI: 0");
  subtitle->setStyleSheet("font-size: 32px; color: #999999;");
  textCol->addWidget(title);
  textCol->addWidget(subtitle);
  hl->addLayout(textCol, 1);

  auto makeBtn = [](const QString &text) {
    QPushButton *b = new QPushButton(text);
    b->setFixedSize(260, 100);
    b->setStyleSheet(R"(
      QPushButton { border-radius: 30px; font-size: 35px; color: #E4E4E4; background-color: #393939; }
      QPushButton:pressed { background-color: #4A4A4A; }
      QPushButton:disabled { color: #33E4E4E4; }
    )");
    return b;
  };
  pairBtn = makeBtn(tr("PAIR"));
  connBtn = makeBtn(tr("CONNECT"));
  trustBtn = makeBtn(tr("TRUST"));
  removeBtn = makeBtn(tr("REMOVE"));

  hl->addWidget(pairBtn);
  hl->addWidget(connBtn);
  hl->addWidget(trustBtn);
  hl->addWidget(removeBtn);

  connect(pairBtn, &QPushButton::clicked, this, [this]() { emit pairRequested(dev.addr); });
  connect(connBtn, &QPushButton::clicked, this, [this]() {
    if (dev.connected) emit disconnectRequested(dev.addr); else emit connectRequested(dev.addr);
  });
  connect(trustBtn, &QPushButton::clicked, this, [this]() { emit trustRequested(dev.addr, !dev.trusted); });
  connect(removeBtn, &QPushButton::clicked, this, [this]() { emit removeRequested(dev.addr); });
}

void DeviceItem::updateButtons() {
  // Pair button enabled if not paired
  pairBtn->setEnabled(!dev.paired);
  // Connect button text toggles by state; only enable when paired
  connBtn->setText(dev.connected ? tr("DISCONNECT") : tr("CONNECT"));
  connBtn->setEnabled(dev.paired);
  // Trust toggle
  trustBtn->setText(dev.trusted ? tr("UNTRUST") : tr("TRUST"));
}

void DeviceItem::setDevice(const BtDevice &d) {
  dev = d;
  QString dn = dev.name.isEmpty() ? tr("(unknown)") : dev.name;
  title->setText(dn);
  QString flags;
  if (dev.paired) flags += tr("paired ");
  if (dev.trusted) flags += tr("trusted ");
  if (dev.connected) flags += tr("connected ");
  flags = flags.trimmed();
  QString sub = QString("%1    •    RSSI: %2%3")
                   .arg(dev.addr)
                   .arg(dev.rssi)
                   .arg(flags.isEmpty() ? "" : QString("    •    %1").arg(flags));
  subtitle->setText(sub);
  updateButtons();
}

BluetoothPanel::BluetoothPanel(QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(50, 20, 50, 20);
  main_layout->setSpacing(30);

  QPushButton *backBtn = new QPushButton(tr("Back"));
  backBtn->setObjectName("back_btn");
  backBtn->setFixedSize(400, 100);
  connect(backBtn, &QPushButton::clicked, this, &BluetoothPanel::backPress);
  main_layout->addWidget(backBtn, 0, Qt::AlignLeft);

  // Title
  QLabel *title = new QLabel(tr("Bluetooth"));
  title->setStyleSheet("font-size: 50px; font-weight: 600;");
  main_layout->addWidget(title, 0, Qt::AlignLeft);

  // Adapter section card
  adapterCard = new QFrame(this);
  adapterCard->setStyleSheet("background-color: #292929; border-radius: 20px; padding: 25px;");
  QVBoxLayout *adap_v = new QVBoxLayout(adapterCard);
  adap_v->setContentsMargins(0, 0, 0, 0);
  adap_v->setSpacing(15);

  QLabel *adap_hdr = new QLabel(tr("Adapter"));
  adap_hdr->setStyleSheet("font-size: 42px; font-weight: 500;");
  adap_v->addWidget(adap_hdr);

  // Status
  status_label = new LabelControl(tr("Status"), tr("Checking..."));
  adap_v->addWidget(status_label);

  // Master enable (bound to Param)
  ParamControl *enable_toggle = new ParamControl("BluetoothEnabled",
                                                 tr("Enable Bluetooth"),
                                                 tr("Master enable/disable for Bluetooth"),
                                                 "");
  QObject::connect(enable_toggle, &ToggleControl::toggleFlipped, this, [this](bool state) {
    runHelperAsync(QString("set-powered %1").arg(state ? "on" : "off"), 5000, [this](int, const QString &) {
      updateStatus();
    });
  });
  adap_v->addWidget(enable_toggle);

  // Discoverable (bound to Param)
  ParamControl *disc_toggle = new ParamControl("BluetoothDiscoverable",
                                               tr("Discoverable"),
                                               tr("Make device visible for pairing"),
                                               "");
  QObject::connect(disc_toggle, &ToggleControl::toggleFlipped, this, [this](bool state) {
    runHelperAsync(QString("set-discoverable %1 --timeout-sec 120").arg(state ? "on" : "off"), 6000, [this](int, const QString &) {
      updateStatus();
    });
  });
  adap_v->addWidget(disc_toggle);

  main_layout->addWidget(adapterCard);

  // Action buttons
  QPushButton *enableNow = new QPushButton(tr("Enable Now"));
  enableNow->setFixedSize(500, 100);
  connect(enableNow, &QPushButton::clicked, [this]() {
    // Run the on-device script to install/enable as needed (async)
    showMessage("");
    runShellAsync("tools/chauffeur/bluetooth3x/check_and_enable_bt.sh", 120000, [this](int code, const QString &) {
      if (code == 0) {
        showMessage(tr("Bluetooth enabled/setup complete"), false);
      } else {
        showMessage(tr("Enable script failed (code %1)").arg(code), true);
      }
      updateStatus();
    });
  });
  main_layout->addWidget(enableNow, 0, Qt::AlignLeft);

  // Devices section card
  devicesCard = new QFrame(this);
  devicesCard->setStyleSheet("background-color: #292929; border-radius: 20px; padding: 25px;");
  QVBoxLayout *dev_v = new QVBoxLayout(devicesCard);
  dev_v->setContentsMargins(0, 0, 0, 0);
  dev_v->setSpacing(15);

  QLabel *dev_hdr = new QLabel(tr("Devices"));
  dev_hdr->setStyleSheet("font-size: 42px; font-weight: 500;");
  dev_v->addWidget(dev_hdr);

  scanBtn = new QPushButton(tr("Scan (once)"));
  scanBtn->setFixedSize(500, 100);
  connect(scanBtn, &QPushButton::clicked, this, &BluetoothPanel::startScanOnce);
  dev_v->addWidget(scanBtn, 0, Qt::AlignLeft);

  scanStatus = new QLabel(tr("Tap Scan to search"));
  scanStatus->setStyleSheet("font-size: 36px; color: #999999;");
  dev_v->addWidget(scanStatus, 0, Qt::AlignLeft);

  msgLabel = new QLabel("");
  msgLabel->setStyleSheet("font-size: 36px; color: #999999;");
  dev_v->addWidget(msgLabel, 0, Qt::AlignLeft);

  device_list_widget = new ListWidget(this);
  dev_v->addWidget(device_list_widget);
  main_layout->addWidget(devicesCard);

  main_layout->addStretch(1);

  // Poll service status periodically
  poll_timer = new QTimer(this);
  poll_timer->setInterval(2000);
  connect(poll_timer, &QTimer::timeout, this, &BluetoothPanel::updateStatus);
  poll_timer->start();
  updateStatus();

  // Style
  setStyleSheet(R"(
    #back_btn {
      font-size: 50px;
      margin: 0px;
      padding: 15px;
      border-width: 0;
      border-radius: 30px;
      color: #dddddd;
      background-color: #393939;
    }
    #back_btn:pressed { background-color: #4a4a4a; }
  )");
}

void BluetoothPanel::updateStatus() {
  if (statusPending) return;
  statusPending = true;
  runHelperAsync("status --json", 2000, [this](int code, const QString &out) {
    QString s;
    if (code == 0) {
      QJsonParseError err;
      QJsonDocument doc = QJsonDocument::fromJson(out.toUtf8(), &err);
      if (err.error == QJsonParseError::NoError && doc.isObject()) {
        QJsonObject o = doc.object();
        bool present = o.value("present").toBool();
        bool powered = o.value("Powered").toBool();
        bool discoverable = o.value("Discoverable").toBool();
        QString addr = o.value("Address").toString();
        s = tr("%1  powered:%2  discoverable:%3").arg(present ? addr : "no controller",
                                                     powered ? "on" : "off",
                                                     discoverable ? "on" : "off");
      }
    }
    if (s.isEmpty()) s = tr("Status unavailable");
    status_label->setText(s);
    statusPending = false;
  });
}

int BluetoothPanel::runHelper(const QString &args, int timeout_ms, QString *stdout_str) {
  QProcess p;
  QString cmd = QString("python3 tools/chauffeur/bluetooth3x/bt_helper.py %1").arg(args);
  p.start("bash", {"-lc", cmd});
  if (!p.waitForFinished(timeout_ms)) {
    p.kill();
    p.waitForFinished(250);
    return -1;
  }
  if (stdout_str) *stdout_str = QString::fromUtf8(p.readAllStandardOutput());
  return p.exitCode();
}

void BluetoothPanel::runHelperAsync(const QString &args, int timeout_ms, std::function<void(int, const QString &)> cb) {
  QProcess *proc = new QProcess(this);
  QString cmd = QString("python3 tools/chauffeur/bluetooth3x/bt_helper.py %1").arg(args);
  QTimer *killer = new QTimer(proc);
  killer->setSingleShot(true);
  connect(killer, &QTimer::timeout, proc, [proc]() {
    if (proc->state() != QProcess::NotRunning) proc->kill();
  });
  connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [proc, killer, cb](int exitCode, QProcess::ExitStatus) {
    QString out = QString::fromUtf8(proc->readAllStandardOutput());
    if (killer) killer->stop();
    proc->deleteLater();
    if (cb) cb(exitCode, out);
  });
  proc->start("bash", {"-lc", cmd});
  killer->start(timeout_ms);
}

void BluetoothPanel::runShellAsync(const QString &cmd, int timeout_ms, std::function<void(int, const QString &)> cb) {
  QProcess *proc = new QProcess(this);
  QTimer *killer = new QTimer(proc);
  killer->setSingleShot(true);
  connect(killer, &QTimer::timeout, proc, [proc]() {
    if (proc->state() != QProcess::NotRunning) proc->kill();
  });
  connect(proc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [proc, killer, cb](int exitCode, QProcess::ExitStatus) {
    QString out = QString::fromUtf8(proc->readAllStandardOutput());
    if (killer) killer->stop();
    proc->deleteLater();
    if (cb) cb(exitCode, out);
  });
  proc->start("bash", {"-lc", cmd});
  killer->start(timeout_ms);
}

void BluetoothPanel::showMessage(const QString &text, bool is_error) {
  if (!msgLabel) return;
  if (text.isEmpty()) {
    msgLabel->clear();
    return;
  }
  msgLabel->setStyleSheet(QString("font-size: 36px; color: %1;").arg(is_error ? "#FFC107" : "#999999"));
  msgLabel->setText(text);
}

void BluetoothPanel::startScanOnce() {
  if (scanProc) {
    return; // already scanning
  }
  // clear message and show progress
  if (msgLabel) msgLabel->clear();
  if (scanBtn) scanBtn->setEnabled(false);
  if (scanStatus) scanStatus->setText(tr("Scanning..."));
  scanProc = new QProcess(this);
  QString cmd = "python3 tools/chauffeur/bluetooth3x/bt_helper.py scan once --timeout 8 --with-info --json";
  connect(scanProc, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, &BluetoothPanel::onScanFinished);
  scanProc->start("bash", {"-lc", cmd});
  if (!scanKillTimer) {
    scanKillTimer = new QTimer(this);
    scanKillTimer->setSingleShot(true);
    connect(scanKillTimer, &QTimer::timeout, this, [this]() {
      if (scanProc && scanProc->state() != QProcess::NotRunning) {
        scanProc->kill();
      }
    });
  }
  scanKillTimer->start(12000);
}

void BluetoothPanel::onScanFinished(int exitCode, QProcess::ExitStatus status) {
  if (scanBtn) scanBtn->setEnabled(true);
  if (scanStatus) scanStatus->setText(tr("Tap Scan to search"));
  if (scanKillTimer) scanKillTimer->stop();
  if (!scanProc) return;
  QString out = QString::fromUtf8(scanProc->readAllStandardOutput());
  scanProc->deleteLater();
  scanProc = nullptr;
  if (exitCode == 0) {
    refreshDevicesFromJson(out);
    if (msgLabel) {
      msgLabel->setStyleSheet("font-size: 36px; color: #999999;");
      msgLabel->setText(tr("Scan complete"));
    }
  } else if (exitCode == 2) {
    if (msgLabel) {
      msgLabel->setStyleSheet("font-size: 36px; color: #FFC107;");
      msgLabel->setText(tr("No controller detected. Tap Enable Now or check rfkill."));
    }
  } else if (exitCode == 3) {
    if (msgLabel) {
      msgLabel->setStyleSheet("font-size: 36px; color: #FFC107;");
      msgLabel->setText(tr("Scan timed out. Try again."));
    }
  } else {
    if (msgLabel) {
      msgLabel->setStyleSheet("font-size: 36px; color: #FFC107;");
      msgLabel->setText(tr("Scan failed (code %1)").arg(exitCode));
    }
  }
}

void BluetoothPanel::refreshDevicesFromJson(const QString &json) {
  devices.clear();
  QJsonParseError err;
  QJsonDocument doc = QJsonDocument::fromJson(json.toUtf8(), &err);
  if (err.error != QJsonParseError::NoError || !doc.isArray()) {
    rebuildDeviceList();
    return;
  }
  QJsonArray arr = doc.array();
  for (auto v : arr) {
    if (!v.isObject()) continue;
    QJsonObject o = v.toObject();
    BtDevice d;
    d.addr = o.value("addr").toString();
    d.name = o.value("name").toString();
    d.rssi = o.value("rssi").toInt(0);
    d.paired = o.value("paired").toBool(false);
    d.trusted = o.value("trusted").toBool(false);
    d.connected = o.value("connected").toBool(false);
    if (!d.addr.isEmpty()) devices.push_back(d);
  }
  // Sort: connected first, then paired, then RSSI desc
  std::sort(devices.begin(), devices.end(), [](const BtDevice &a, const BtDevice &b) {
    if (a.connected != b.connected) return a.connected; // true first
    if (a.paired != b.paired) return a.paired; // true first
    return a.rssi > b.rssi;
  });
  rebuildDeviceList();
}

void BluetoothPanel::rebuildDeviceList() {
  // clear existing widgets
  if (!device_list_widget) return;
  auto layout = device_list_widget->layout();
  if (layout) {
    // remove all widgets from inner layout of ListWidget
    // cannot access inner_layout directly; instead, delete children of device_list_widget
    QList<QObject*> children = device_list_widget->children();
    for (QObject *obj : children) {
      QWidget *w = qobject_cast<QWidget*>(obj);
      if (w) w->deleteLater();
    }
  }
  // Recreate a fresh ListWidget: simpler and safe
  QVBoxLayout *dev_v = qobject_cast<QVBoxLayout*>(devicesCard->layout());
  if (dev_v) {
    dev_v->removeWidget(device_list_widget);
    device_list_widget->deleteLater();
    device_list_widget = new ListWidget(this);
    dev_v->addWidget(device_list_widget);
  }

  if (devices.isEmpty()) {
    QLabel *empty = new QLabel(tr("No devices. Tap Scan."));
    empty->setStyleSheet("font-size: 36px; color: #999999;");
    device_list_widget->addItem(new LayoutWidget(new QHBoxLayout)); // spacer line
    device_list_widget->addItem(empty);
    return;
  }

  for (const auto &d : devices) {
    DeviceItem *item = new DeviceItem(this);
    item->setDevice(d);
    connect(item, &DeviceItem::pairRequested, this, [this](const QString &addr) {
      runHelperAsync(QString("pair-interactive %1 --timeout 30 --json").arg(addr), 35000, [this, addr](int rc, const QString &out) {
        if (rc == 0) {
          if (msgLabel) { msgLabel->setStyleSheet("font-size: 36px; color: #999999;"); msgLabel->setText(tr("Paired successfully")); }
          startScanOnce();
          return;
        }
        if (rc == 10 || rc == 11 || rc == 12) {
          QJsonParseError err; QJsonDocument doc = QJsonDocument::fromJson(out.toUtf8(), &err);
          QString event = (err.error == QJsonParseError::NoError && doc.isObject()) ? doc.object().value("event").toString() : QString();
          if (rc == 10 || event == "confirm") {
            QString passkey = (doc.isObject() ? doc.object().value("passkey").toString() : QString());
            QString content = tr("<body><h2 style=\"text-align:center;\">Confirm Passkey</h2><br><p style=\"text-align:center; font-size: 50px;\">%1</p></body>").arg(passkey);
            bool yes = ConfirmationDialog::rich(content, this);
            runHelperAsync(QString("pair-complete %1 --confirm %2").arg(addr).arg(yes ? "yes" : "no"), 35000, [this](int rc2, const QString &) {
              if (rc2 != 0) {
                if (msgLabel) { msgLabel->setStyleSheet("font-size: 36px; color: #FFC107;"); msgLabel->setText(tr("Pair confirm failed (code %1)").arg(rc2)); }
              } else {
                if (msgLabel) { msgLabel->setStyleSheet("font-size: 36px; color: #999999;"); msgLabel->setText(tr("Paired successfully")); }
              }
              startScanOnce();
            });
            return;
          } else if (rc == 11 || event == "pin") {
            QString pin = InputDialog::getText(tr("Enter PIN"), this, tr("for %1").arg(addr), false, 1);
            if (pin.isEmpty()) return;
            runHelperAsync(QString("pair-complete %1 --pin %2").arg(addr).arg(pin), 35000, [this](int rc2, const QString &) {
              if (rc2 != 0) {
                if (msgLabel) { msgLabel->setStyleSheet("font-size: 36px; color: #FFC107;"); msgLabel->setText(tr("Pair (PIN) failed (code %1)").arg(rc2)); }
              } else {
                if (msgLabel) { msgLabel->setStyleSheet("font-size: 36px; color: #999999;"); msgLabel->setText(tr("Paired successfully")); }
              }
              startScanOnce();
            });
            return;
          } else if (rc == 12 || event == "passkey") {
            QString pk = InputDialog::getText(tr("Enter passkey"), this, tr("for %1").arg(addr), false, 1);
            if (pk.isEmpty()) return;
            runHelperAsync(QString("pair-complete %1 --passkey %2").arg(addr).arg(pk), 35000, [this](int rc2, const QString &) {
              if (rc2 != 0) {
                if (msgLabel) { msgLabel->setStyleSheet("font-size: 36px; color: #FFC107;"); msgLabel->setText(tr("Pair (passkey) failed (code %1)").arg(rc2)); }
              } else {
                if (msgLabel) { msgLabel->setStyleSheet("font-size: 36px; color: #999999;"); msgLabel->setText(tr("Paired successfully")); }
              }
              startScanOnce();
            });
            return;
          }
        }
        if (msgLabel) { msgLabel->setStyleSheet("font-size: 36px; color: #FFC107;"); msgLabel->setText(tr("Pair failed (code %1)").arg(rc)); }
        startScanOnce();
      });
    });
    connect(item, &DeviceItem::connectRequested, this, [this](const QString &addr) {
      runHelperAsync(QString("connect %1").arg(addr), 10000, [this](int rc, const QString &) {
        if (rc != 0) {
          if (msgLabel) { msgLabel->setStyleSheet("font-size: 36px; color: #FFC107;"); msgLabel->setText(tr("Connect failed (code %1)").arg(rc)); }
        }
        startScanOnce();
      });
    });
    connect(item, &DeviceItem::disconnectRequested, this, [this](const QString &addr) {
      runHelperAsync(QString("disconnect %1").arg(addr), 8000, [this](int rc, const QString &) {
        if (rc != 0) {
          if (msgLabel) { msgLabel->setStyleSheet("font-size: 36px; color: #FFC107;"); msgLabel->setText(tr("Disconnect failed (code %1)").arg(rc)); }
        }
        startScanOnce();
      });
    });
    connect(item, &DeviceItem::trustRequested, this, [this](const QString &addr, bool trust) {
      runHelperAsync(QString("trust %1 %2").arg(addr).arg(trust ? "on" : "off"), 8000, [this](int rc, const QString &) {
        if (rc != 0) {
          if (msgLabel) { msgLabel->setStyleSheet("font-size: 36px; color: #FFC107;"); msgLabel->setText(tr("Trust toggle failed (code %1)").arg(rc)); }
        }
        startScanOnce();
      });
    });
    connect(item, &DeviceItem::removeRequested, this, [this](const QString &addr) {
      runHelperAsync(QString("remove %1").arg(addr), 8000, [this](int rc, const QString &) {
        if (rc != 0) {
          if (msgLabel) { msgLabel->setStyleSheet("font-size: 36px; color: #FFC107;"); msgLabel->setText(tr("Remove failed (code %1)").arg(rc)); }
        }
        startScanOnce();
      });
    });
    device_list_widget->addItem(item);
  }
}
