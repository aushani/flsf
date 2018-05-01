#pragma once

#include <boost/filesystem.hpp>

#include "library/flow/flow_processor.h"
#include "library/flow/distance_map.h"
#include "library/kitti/velodyne_scan.h"
#include "library/kitti/nodes/point_cloud.h"
#include "library/kitti/nodes/tracklets.h"
#include "library/flow/nodes/flow_image.h"
//#include "library/flow/nodes/classification_map.h"
#include "library/flow/nodes/filter_map.h"
#include "library/osg_nodes/car.h"
#include "library/ray_tracing/nodes/occ_grid.h"
#include "library/flow/nodes/distance_map.h"
#include "library/viewer/viewer.h"

namespace fs = boost::filesystem;

namespace fl = library::flow;
namespace kt = library::kitti;
namespace vw = library::viewer;
namespace osgn = library::osg_nodes;
namespace rtn = library::ray_tracing::nodes;
namespace fln = library::flow::nodes;
namespace ktn = library::kitti::nodes;

namespace app {
namespace flow {

class NodeManager {
 public:
  NodeManager(const fs::path &car_path);

  void SetViewer(const std::shared_ptr<vw::Viewer> &viewer);

  void Update(const fl::FlowProcessor &fp, const kt::VelodyneScan &scan1, const kt::VelodyneScan &scan2, kt::Tracklets *tracklets, int frame_num);
  void ShowDistanceMap(const fl::FlowProcessor &fp, double x, double y);
  void ClearDistanceMap();

  void SetViewMode(int view_mode);

 private:
  int view_mode_ = 1;

  std::shared_ptr<vw::Viewer> viewer_;

  osg::ref_ptr<ktn::PointCloud> pc1_;
  osg::ref_ptr<ktn::PointCloud> pc2_;

  osg::ref_ptr<ktn::Tracklets> tn1_;
  osg::ref_ptr<ktn::Tracklets> tn2_;

  osg::ref_ptr<rtn::OccGrid> og1n_;
  osg::ref_ptr<rtn::OccGrid> og2n_;

  osg::ref_ptr<fln::FlowImage> fin_;
  //osg::ref_ptr<fln::ClassificationMap> cmn_;
  osg::ref_ptr<fln::FilterMap> fmn_;

  osg::ref_ptr<fln::DistanceMap> dmn_;

  osg::ref_ptr<osgn::Car> car_node_;

  void UpdateViewer();
};

} // flow
} // app
