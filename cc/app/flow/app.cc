#include "app/flow/app.h"

#include <boost/format.hpp>

namespace app {
namespace flow {

App::App(const fs::path &tsf_dir, const std::string &date, int log_num) :
 og_builder_(kMaxVelodyneScanPoints, kResolution, kMaxRange) {

  // Load data
  std::string dir_name = (boost::format("%s_drive_%04d_sync") % date % log_num).str();

  printf("Loading velodyne data...\n");
  int frame_num = 0;
  while (true) {
    std::string fn = (boost::format("%010d.bin") % frame_num).str();
    fs::path path = tsf_dir / "kittidata" / date / dir_name / "velodyne_points" / "data" / fn;

    if (!fs::exists(path)) {
      break;
    }

    //printf("Load %s\n", path.string().c_str());
    scans_.emplace_back(path.string());

    frame_num++;
  }

  printf("Loading tracklet data...\n");
  fs::path tracklet_file = tsf_dir / "kittidata" / date / dir_name / "tracklet_labels.xml";
  tracklets_.loadFromFile(tracklet_file.string());
}

} // flow
} // app

