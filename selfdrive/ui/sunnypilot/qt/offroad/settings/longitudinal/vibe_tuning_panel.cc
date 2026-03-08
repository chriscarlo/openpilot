/**
 * Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.
 *
 * This file is part of sunnypilot and is licensed under the MIT License.
 * See the LICENSE.md file in the root directory for more details.
 */

#include "selfdrive/ui/sunnypilot/qt/offroad/settings/longitudinal/vibe_tuning_panel.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QVector>
#include <QWidget>

#include "common/params.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"

namespace {

struct FloatParamDescriptor {
  QString title;
  QString description;
  QString key;
  float min_value;
  float max_value;
  float step;
  float default_value;
  QString units;
  int decimals;
};

struct SectionDescriptor {
  QString title;
  QString description;
  QVector<FloatParamDescriptor> controls;
};

struct BankDescriptor {
  QString bank_id;
  QString title;
  QString subtitle;
  QString hub_title;
  QString hub_description;
  QVector<SectionDescriptor> sections;
};

QString makeVibeParamKey(const QString &group, const QString &mode, const QString &field, int index) {
  return QString("VibeTune.%1.%2.%3%4").arg(group, mode, field).arg(index);
}

QFrame *createSectionFrame(QWidget *parent = nullptr) {
  QFrame *frame = new QFrame(parent);
  frame->setStyleSheet("QFrame { background-color: #292929; border-radius: 20px; padding: 25px; }");
  return frame;
}

class VibeFloatRangeControl : public QFrame {
public:
  VibeFloatRangeControl(const FloatParamDescriptor &descriptor, QWidget *parent = nullptr)
    : QFrame(parent), descriptor_(descriptor) {
    QVBoxLayout *main_layout = new QVBoxLayout(this);
    main_layout->setContentsMargins(0, 0, 0, 0);

    QLabel *title_label = new QLabel(descriptor_.title);
    title_label->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4;");
    main_layout->addWidget(title_label);

    if (!descriptor_.description.isEmpty()) {
      QLabel *description_label = new QLabel(descriptor_.description);
      description_label->setWordWrap(true);
      description_label->setStyleSheet("font-size: 32px; color: #999999; margin-top: 5px; margin-bottom: 15px;");
      main_layout->addWidget(description_label);
    }

    QHBoxLayout *control_layout = new QHBoxLayout();
    control_layout->setSpacing(20);

    minus_btn_ = new QPushButton("-");
    minus_btn_->setFixedSize(100, 100);
    minus_btn_->setStyleSheet(R"(
      QPushButton {
        font-size: 60px;
        font-weight: 500;
        border-radius: 50px;
        background-color: #393939;
        color: #E4E4E4;
      }
      QPushButton:pressed {
        background-color: #4a4a4a;
      }
      QPushButton:disabled {
        background-color: #2a2a2a;
        color: #666666;
      }
    )");
    minus_btn_->setFocusPolicy(Qt::NoFocus);
    control_layout->addWidget(minus_btn_);

    QVBoxLayout *value_layout = new QVBoxLayout();
    value_layout->setAlignment(Qt::AlignCenter);

    value_label_ = new QLabel();
    value_label_->setAlignment(Qt::AlignCenter);
    value_label_->setStyleSheet("font-size: 70px; font-weight: 500; color: #FFFFFF;");
    value_label_->setFixedWidth(340);
    value_layout->addWidget(value_label_);

    status_label_ = new QLabel();
    status_label_->setAlignment(Qt::AlignCenter);
    status_label_->setStyleSheet("font-size: 32px; color: #999999;");
    value_layout->addWidget(status_label_);

    control_layout->addLayout(value_layout);

    plus_btn_ = new QPushButton("+");
    plus_btn_->setFixedSize(100, 100);
    plus_btn_->setStyleSheet(minus_btn_->styleSheet());
    plus_btn_->setFocusPolicy(Qt::NoFocus);
    control_layout->addWidget(plus_btn_);

    control_layout->addStretch();

    reset_btn_ = new QPushButton(QObject::tr("Reset"));
    reset_btn_->setFixedSize(150, 80);
    reset_btn_->setStyleSheet(R"(
      QPushButton {
        font-size: 35px;
        font-weight: 500;
        border-radius: 20px;
        background-color: #393939;
        color: #E4E4E4;
      }
      QPushButton:pressed {
        background-color: #4a4a4a;
      }
      QPushButton:disabled {
        background-color: #2a2a2a;
        color: #666666;
      }
    )");
    reset_btn_->setFocusPolicy(Qt::NoFocus);
    control_layout->addWidget(reset_btn_);

    main_layout->addLayout(control_layout);

    connect(minus_btn_, &QPushButton::clicked, this, [this]() { adjustValue(-descriptor_.step); });
    connect(plus_btn_, &QPushButton::clicked, this, [this]() { adjustValue(descriptor_.step); });
    connect(reset_btn_, &QPushButton::clicked, this, [this]() { setValue(descriptor_.default_value); });

    refresh();
  }

  void refresh() {
    current_value_ = loadStoredValue();
    updateLabels();
  }

private:
  float loadStoredValue() {
    QString stored_value = QString::fromStdString(params_.get(descriptor_.key.toStdString()));
    if (stored_value.isEmpty()) {
      return descriptor_.default_value;
    }

    bool ok = false;
    const float parsed = stored_value.toFloat(&ok);
    return ok ? parsed : descriptor_.default_value;
  }

  void adjustValue(float delta) {
    setValue(current_value_ + delta);
  }

  void setValue(float value) {
    current_value_ = std::clamp(value, descriptor_.min_value, descriptor_.max_value);
    const float scale = std::pow(10.0f, static_cast<float>(descriptor_.decimals));
    current_value_ = std::round(current_value_ * scale) / scale;
    params_.put(descriptor_.key.toStdString(), QString::number(current_value_, 'f', descriptor_.decimals).toStdString());
    updateLabels();
  }

  void updateLabels() {
    value_label_->setText(QString("%1 %2").arg(QString::number(current_value_, 'f', descriptor_.decimals), descriptor_.units));

    const bool is_default = std::abs(current_value_ - descriptor_.default_value) < 0.0005f;
    if (is_default) {
      status_label_->setText(QObject::tr("(Default)"));
      status_label_->setStyleSheet("font-size: 32px; color: #999999;");
    } else {
      status_label_->setText(QObject::tr("(Modified)"));
      status_label_->setStyleSheet("font-size: 32px; color: #FFC107;");
    }

    minus_btn_->setEnabled(current_value_ > descriptor_.min_value + 0.0005f);
    plus_btn_->setEnabled(current_value_ < descriptor_.max_value - 0.0005f);
    reset_btn_->setEnabled(!is_default);
  }

  Params params_;
  FloatParamDescriptor descriptor_;
  float current_value_ = 0.0f;
  QLabel *value_label_ = nullptr;
  QLabel *status_label_ = nullptr;
  QPushButton *minus_btn_ = nullptr;
  QPushButton *plus_btn_ = nullptr;
  QPushButton *reset_btn_ = nullptr;
};

class VibeBankScreen : public QWidget {
public:
  explicit VibeBankScreen(const BankDescriptor &descriptor, std::function<void()> on_back, QWidget *parent = nullptr)
    : QWidget(parent), descriptor_(descriptor), on_back_(std::move(on_back)) {
    QVBoxLayout *main_layout = new QVBoxLayout(this);
    main_layout->setContentsMargins(50, 20, 50, 20);
    main_layout->setSpacing(30);

    PanelBackButton *back_btn = new PanelBackButton(QObject::tr("Back"));
    connect(back_btn, &QPushButton::clicked, this, [this]() {
      if (on_back_) {
        on_back_();
      }
    });
    main_layout->addWidget(back_btn, 0, Qt::AlignLeft);

    main_layout->addSpacing(20);

    QLabel *title_label = new QLabel(descriptor_.title);
    title_label->setAlignment(Qt::AlignCenter);
    title_label->setStyleSheet("font-size: 50px; font-weight: 600; color: #E4E4E4; padding-bottom: 10px;");
    main_layout->addWidget(title_label);

    QLabel *subtitle_label = new QLabel(descriptor_.subtitle);
    subtitle_label->setAlignment(Qt::AlignCenter);
    subtitle_label->setWordWrap(true);
    subtitle_label->setStyleSheet("font-size: 34px; color: #999999; padding-bottom: 20px;");
    main_layout->addWidget(subtitle_label);

    QScrollArea *scroll_area = new QScrollArea(this);
    scroll_area->setWidgetResizable(true);
    scroll_area->setFrameShape(QFrame::NoFrame);
    scroll_area->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll_area->setStyleSheet("QScrollArea { background: transparent; }");

    QWidget *content = new QWidget(scroll_area);
    QVBoxLayout *content_layout = new QVBoxLayout(content);
    content_layout->setContentsMargins(0, 0, 0, 20);
    content_layout->setSpacing(30);

    for (const SectionDescriptor &section : descriptor_.sections) {
      QFrame *section_frame = createSectionFrame(content);
      QVBoxLayout *section_layout = new QVBoxLayout(section_frame);

      QLabel *section_label = new QLabel(section.title);
      section_label->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
      section_layout->addWidget(section_label);

      if (!section.description.isEmpty()) {
        QLabel *section_description = new QLabel(section.description);
        section_description->setWordWrap(true);
        section_description->setStyleSheet("font-size: 32px; color: #999999; padding-bottom: 20px;");
        section_layout->addWidget(section_description);
      }

      bool first = true;
      for (const FloatParamDescriptor &control_descriptor : section.controls) {
        if (!first) {
          section_layout->addSpacing(20);
        }
        first = false;
        VibeFloatRangeControl *control = new VibeFloatRangeControl(control_descriptor, section_frame);
        range_controls_.append(control);
        section_layout->addWidget(control);
      }

      content_layout->addWidget(section_frame);
    }

    content_layout->addStretch();
    scroll_area->setWidget(content);
    main_layout->addWidget(scroll_area, 1);
  }

  void refresh() {
    for (VibeFloatRangeControl *control : range_controls_) {
      if (control != nullptr) {
        control->refresh();
      }
    }
  }

private:
  BankDescriptor descriptor_;
  std::function<void()> on_back_;
  QVector<VibeFloatRangeControl *> range_controls_;
};

QVector<FloatParamDescriptor> makeFollowHeadwayControls(const QString &mode_name, const std::initializer_list<float> &defaults) {
  const QVector<QString> titles = {
    QObject::tr("Headway @ 0 mph"),
    QObject::tr("Headway @ 45 mph"),
    QObject::tr("Headway @ 50 mph"),
    QObject::tr("Headway @ 90 mph"),
  };
  const QVector<QString> descriptions = {
    QObject::tr("Desired time gap from a stopped or crawling launch."),
    QObject::tr("Desired time gap around the first freeway merge breakpoint."),
    QObject::tr("Desired time gap near the second freeway breakpoint."),
    QObject::tr("Desired time gap used at high freeway speed and above."),
  };

  QVector<FloatParamDescriptor> controls;
  int index = 0;
  for (float default_value : defaults) {
    controls.push_back({
      titles[index],
      descriptions[index],
      makeVibeParamKey("Follow", mode_name, "Headway", index),
      0.80f,
      3.00f,
      0.01f,
      default_value,
      "s",
      2,
    });
    ++index;
  }
  return controls;
}

QVector<FloatParamDescriptor> makeBrakeControls(const QString &mode_name, const std::initializer_list<float> &defaults) {
  const QVector<QString> titles = {
    QObject::tr("Brake Floor @ 0 mph"),
    QObject::tr("Brake Floor @ 22 mph"),
    QObject::tr("Brake Floor @ 56 mph"),
    QObject::tr("Brake Floor @ 112+ mph"),
  };
  const QVector<QString> descriptions = {
    QObject::tr("Minimum response-model decel at launch and crawl speeds. More negative is firmer."),
    QObject::tr("Minimum response-model decel in low-speed city driving."),
    QObject::tr("Minimum response-model decel at freeway speed."),
    QObject::tr("Minimum response-model decel for the top-end anchor."),
  };

  QVector<FloatParamDescriptor> controls;
  int index = 0;
  for (float default_value : defaults) {
    controls.push_back({
      titles[index],
      descriptions[index],
      makeVibeParamKey("Brake", mode_name, "Decel", index),
      -3.00f,
      -0.20f,
      0.05f,
      default_value,
      "m/s^2",
      2,
    });
    ++index;
  }
  return controls;
}

QVector<FloatParamDescriptor> makeAccelControls(const QString &mode_name, const std::initializer_list<float> &defaults) {
  const QVector<QString> titles = {
    QObject::tr("Max Accel @ 0 mph"),
    QObject::tr("Max Accel @ 13 mph"),
    QObject::tr("Max Accel @ 20 mph"),
    QObject::tr("Max Accel @ 25 mph"),
    QObject::tr("Max Accel @ 36 mph"),
    QObject::tr("Max Accel @ 45 mph"),
    QObject::tr("Max Accel @ 56 mph"),
    QObject::tr("Max Accel @ 67 mph"),
    QObject::tr("Max Accel @ 123+ mph"),
  };
  const QVector<QString> descriptions = {
    QObject::tr("Launch acceleration cap from standstill."),
    QObject::tr("Acceleration cap in neighborhood driving."),
    QObject::tr("Acceleration cap in light city traffic."),
    QObject::tr("Acceleration cap in brisk city traffic."),
    QObject::tr("Acceleration cap on lower-speed arterials."),
    QObject::tr("Acceleration cap near typical urban freeway speed."),
    QObject::tr("Acceleration cap near highway cruise speed."),
    QObject::tr("Acceleration cap at fast freeway speed."),
    QObject::tr("Acceleration cap for the top-end anchor and above."),
  };

  QVector<FloatParamDescriptor> controls;
  int index = 0;
  for (float default_value : defaults) {
    controls.push_back({
      titles[index],
      descriptions[index],
      makeVibeParamKey("Accel", mode_name, "Max", index),
      0.10f,
      5.00f,
      0.01f,
      default_value,
      "m/s^2",
      2,
    });
    ++index;
  }
  return controls;
}

QVector<BankDescriptor> buildBankDescriptors() {
  return {
    {
      "FollowAggressive",
      QObject::tr("Aggressive Driving Bank"),
      QObject::tr("Tune the aggressive Vibe follow-distance and braking anchors used when Driving Personality is set to Aggressive."),
      QObject::tr("Aggressive Driving"),
      QObject::tr("Tune aggressive-mode headway and braking anchors."),
      {
        {
          QObject::tr("Headway"),
          QObject::tr("Shorter gaps close traffic more tightly. Higher values keep more following distance."),
          makeFollowHeadwayControls("Aggressive", {1.20f, 1.20f, 1.30f, 1.30f}),
        },
        {
          QObject::tr("Braking"),
          QObject::tr("More negative floors allow firmer decel planning when Vibe acceleration tuning is enabled."),
          makeBrakeControls("Aggressive", {-1.10f, -1.25f, -1.40f, -1.40f}),
        },
      },
    },
    {
      "FollowStandard",
      QObject::tr("Standard Driving Bank"),
      QObject::tr("Tune the standard Vibe follow-distance and braking anchors used when Driving Personality is set to Standard."),
      QObject::tr("Standard Driving"),
      QObject::tr("Tune standard-mode headway and braking anchors."),
      {
        {
          QObject::tr("Headway"),
          QObject::tr("Balanced follow-distance anchors for standard driving."),
          makeFollowHeadwayControls("Standard", {1.35f, 1.35f, 1.40f, 1.40f}),
        },
        {
          QObject::tr("Braking"),
          QObject::tr("Balanced response-model decel floors for standard driving."),
          makeBrakeControls("Standard", {-1.05f, -1.15f, -1.30f, -1.30f}),
        },
      },
    },
    {
      "FollowRelaxed",
      QObject::tr("Relaxed Driving Bank"),
      QObject::tr("Tune the relaxed Vibe follow-distance and braking anchors used when Driving Personality is set to Relaxed."),
      QObject::tr("Relaxed Driving"),
      QObject::tr("Tune relaxed-mode headway and braking anchors."),
      {
        {
          QObject::tr("Headway"),
          QObject::tr("Longer gaps emphasize comfort and extra space, especially at freeway speed."),
          makeFollowHeadwayControls("Relaxed", {1.25f, 1.60f, 1.85f, 2.20f}),
        },
        {
          QObject::tr("Braking"),
          QObject::tr("Gentler response-model decel floors for relaxed driving."),
          makeBrakeControls("Relaxed", {-0.50f, -0.80f, -1.20f, -1.20f}),
        },
      },
    },
    {
      "AccelSport",
      QObject::tr("Sport Acceleration Bank"),
      QObject::tr("Tune the sport Vibe acceleration anchors used when Acceleration Personality is set to Sport."),
      QObject::tr("Sport Acceleration"),
      QObject::tr("Tune sport-mode maximum acceleration anchors."),
      {
        {
          QObject::tr("Acceleration"),
          QObject::tr("Higher caps let the planner command stronger positive acceleration when Vibe acceleration tuning is enabled."),
          makeAccelControls("Sport", {4.00f, 4.00f, 3.80f, 3.50f, 2.00f, 1.75f, 1.325f, 1.15f, 0.50f}),
        },
      },
    },
    {
      "AccelNormal",
      QObject::tr("Normal Acceleration Bank"),
      QObject::tr("Tune the normal Vibe acceleration anchors used when Acceleration Personality is set to Normal."),
      QObject::tr("Normal Acceleration"),
      QObject::tr("Tune normal-mode maximum acceleration anchors."),
      {
        {
          QObject::tr("Acceleration"),
          QObject::tr("Balanced positive-acceleration anchors for Vibe normal mode."),
          makeAccelControls("Normal", {2.00f, 2.00f, 1.42f, 1.10f, 0.65f, 0.56f, 0.43f, 0.36f, 0.12f}),
        },
      },
    },
    {
      "AccelEco",
      QObject::tr("Eco Acceleration Bank"),
      QObject::tr("Tune the eco Vibe acceleration anchors used when Acceleration Personality is set to Eco."),
      QObject::tr("Eco Acceleration"),
      QObject::tr("Tune eco-mode maximum acceleration anchors."),
      {
        {
          QObject::tr("Acceleration"),
          QObject::tr("Lower caps emphasize smoother and less aggressive acceleration."),
          makeAccelControls("Eco", {1.10f, 1.00f, 0.85f, 0.76f, 0.58f, 0.46f, 0.365f, 0.317f, 0.089f}),
        },
      },
    },
  };
}

}  // namespace

VibeTuningPanel::VibeTuningPanel(QWidget *parent) : QFrame(parent) {
  setupUi();
}

void VibeTuningPanel::setupUi() {
  const QVector<BankDescriptor> bank_descriptors = buildBankDescriptors();

  QVBoxLayout *root_layout = new QVBoxLayout(this);
  root_layout->setContentsMargins(0, 0, 0, 0);

  stacked_layout = new QStackedLayout();
  root_layout->addLayout(stacked_layout);

  hub_screen = new QWidget(this);
  stacked_layout->addWidget(hub_screen);

  QVBoxLayout *hub_layout = new QVBoxLayout(hub_screen);
  hub_layout->setContentsMargins(50, 20, 50, 20);
  hub_layout->setSpacing(30);

  PanelBackButton *back_btn = new PanelBackButton(tr("Back"));
  connect(back_btn, &QPushButton::clicked, this, &VibeTuningPanel::backPress);
  hub_layout->addWidget(back_btn, 0, Qt::AlignLeft);

  hub_layout->addSpacing(20);

  QLabel *title_label = new QLabel(tr("Vibe Personality Tuning"));
  title_label->setAlignment(Qt::AlignCenter);
  title_label->setStyleSheet("font-size: 50px; font-weight: 600; color: #E4E4E4; padding-bottom: 10px;");
  hub_layout->addWidget(title_label);

  QLabel *description_label = new QLabel(tr("Choose one of the six Vibe tuning banks to edit the persistent profile anchors used by the controller. Defaults match today’s shipped Vibe tables."));
  description_label->setAlignment(Qt::AlignCenter);
  description_label->setWordWrap(true);
  description_label->setStyleSheet("font-size: 34px; color: #999999; padding-bottom: 20px;");
  hub_layout->addWidget(description_label);

  QFrame *follow_frame = createSectionFrame(hub_screen);
  QVBoxLayout *follow_layout = new QVBoxLayout(follow_frame);
  QLabel *follow_label = new QLabel(tr("Driving Banks"));
  follow_label->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  follow_layout->addWidget(follow_label);
  hub_layout->addWidget(follow_frame);

  QFrame *accel_frame = createSectionFrame(hub_screen);
  QVBoxLayout *accel_layout = new QVBoxLayout(accel_frame);
  QLabel *accel_label = new QLabel(tr("Acceleration Banks"));
  accel_label->setStyleSheet("font-size: 42px; font-weight: 500; color: #E4E4E4; padding-bottom: 15px;");
  accel_layout->addWidget(accel_label);
  hub_layout->addWidget(accel_frame);

  for (const BankDescriptor &descriptor : bank_descriptors) {
    VibeBankScreen *bank_screen = new VibeBankScreen(descriptor, [this]() {
      stacked_layout->setCurrentWidget(hub_screen);
    }, this);
    stacked_layout->addWidget(bank_screen);

    ButtonControlSP *bank_button = new ButtonControlSP(descriptor.hub_title, tr("Open"), descriptor.hub_description, hub_screen);
    connect(bank_button, &ButtonControlSP::clicked, this, [this, bank_screen]() {
      bank_screen->refresh();
      stacked_layout->setCurrentWidget(bank_screen);
    });

    if (descriptor.bank_id.startsWith("Follow")) {
      follow_layout->addWidget(bank_button);
    } else {
      accel_layout->addWidget(bank_button);
    }
  }

  follow_layout->addStretch();
  accel_layout->addStretch();
  hub_layout->addStretch();
}

void VibeTuningPanel::showEvent(QShowEvent *event) {
  QFrame::showEvent(event);
  if (stacked_layout && hub_screen) {
    stacked_layout->setCurrentWidget(hub_screen);
  }
}
