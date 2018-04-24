#include <iostream>
#include <osg/ArgumentParser>

#include "library/viewer/viewer.h"

#include "app/flow/app.h"
#include "app/flow/handler.h"
#include "app/flow/click_handler.h"

namespace vw = library::viewer;

int main(int argc, char** argv) {
  osg::ArgumentParser args(&argc, argv);
  osg::ApplicationUsage* au = args.getApplicationUsage();

  // report any errors if they have occurred when parsing the program arguments.
  if (args.errors()) {
    args.writeErrorMessages(std::cout);
    return EXIT_FAILURE;
  }

  au->setApplicationName(args.getApplicationName());
  au->setCommandLineUsage( args.getApplicationName() + " [options]");
  au->setDescription(args.getApplicationName() + " viewer");

  au->addCommandLineOption("--tsf-data-dir <dirname>", "TSF data directory", "~/data/tsf_data/");
  au->addCommandLineOption("--kitti-log-date <dirname>", "KITTI date", "2011_09_26");
  au->addCommandLineOption("--log-num <num>", "KITTI log number", "18");

  // handle help text
  // call AFTER init viewer so key bindings have been set
  unsigned int helpType = 0;
  if ((helpType = args.readHelpType())) {
    au->write(std::cout, helpType);
    return EXIT_SUCCESS;
  }

  // Read params
  std::string home_dir = getenv("HOME");
  std::string tsf_data_dir = home_dir + "/data/tsf_data";
  if (!args.read("--tsf-data-dir", tsf_data_dir)) {
    printf("Using default tsf data dir: %s\n", tsf_data_dir.c_str());
  }

  std::string kitti_log_date = "2011_09_26";
  if (!args.read("--kitti-log-date", kitti_log_date)) {
    printf("Using default KITTI date: %s\n", kitti_log_date.c_str());
  }

  int log_num = 18;
  if (!args.read("--log-num", log_num)) {
    printf("Using default KITTI log number: %d\n", log_num);
  }

  auto app = std::make_shared<app::flow::App>(fs::path(tsf_data_dir), kitti_log_date, log_num);
  auto viewer = std::make_shared<vw::Viewer>(&args);

  osg::ref_ptr<app::flow::Handler> handler(new app::flow::Handler(app));
  osg::ref_ptr<app::flow::ClickHandler> click_handler(new app::flow::ClickHandler(app));

  viewer->AddHandler(handler);
  viewer->AddHandler(click_handler);
  app->SetViewer(viewer);

  // Process the first frame
  app->ProcessNext();

  // Now we're ready to start
  printf("Ready...\n");
  viewer->Start();

  return 0;
}
