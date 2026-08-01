#include <sys/resource.h>

#include <QApplication>
#include <QTranslator>

#include "common/params.h"
#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/util.h"
#include "selfdrive/ui/qt/window.h"
#include "selfdrive/ui/ui.h"

#ifdef SUNNYPILOT
#include "selfdrive/ui/sunnypilot/qt/window.h"
#define MainWindow MainWindowSP
#else
#include "selfdrive/ui/qt/qt_window.h"
#endif

int main(int argc, char *argv[]) {
  setpriority(PRIO_PROCESS, 0, -20);

  qInstallMessageHandler(swagLogMessageHandler);
  initApp(argc, argv);

  // Bind the process-wide "bookmarkButton" publisher NOW, not on first press.
  // msgq_init_publisher() evicts every attached subscriber (loggerd, feedbackd,
  // plannerd), and they only re-attach on their next poll -- skipping straight
  // to the current write pointer. Publishing on the same call that binds the
  // endpoint therefore drops the message. See selfdrive/ui/ui.h. This is the
  // same construction point the pre-refactor Sidebar ctor effectively had, and
  // it covers both the stock and SUNNYPILOT builds (main.cc is in qt_src for
  // both, see selfdrive/ui/SConscript).
  initBookmarkPublisher();

  QTranslator translator;
  QString translation_file = QString::fromStdString(Params().get("LanguageSetting"));
  if (!translator.load(QString(":/%1").arg(translation_file)) && translation_file.length()) {
    qCritical() << "Failed to load translation file:" << translation_file;
  }

  QApplication a(argc, argv);
  a.installTranslator(&translator);

  MainWindow w;
  setMainWindow(&w);
  a.installEventFilter(&w);
  return a.exec();
}
