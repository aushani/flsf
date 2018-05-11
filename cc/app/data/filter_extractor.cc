#include "app/data/filter_extractor.h"

#include "library/ray_tracing/occ_grid_location.h"

#include "library/kitti/util.h"

namespace app {
namespace data {

FilterExtractor::FilterExtractor(const fs::path &base_path, const fs::path &save_path) :
 scans_(kt::VelodyneScan::LoadDirectory(base_path / "velodyne_points" / "data")),
 camera_cal_(base_path.parent_path()),
 og_builder_(ps::kMaxVelodyneScanPoints, ps::kResolution, ps::kMaxRange),
 save_file_( (save_path / "filter.bin").string(), std::ios::binary) {
  tracklets_.loadFromFile( (base_path / "tracklet_labels.xml").string() );
}

void FilterExtractor::WriteOccGrid(const rt::OccGrid &og) {
  // Get bounds
  int i0 = ps::kOccGridMinXY;
  int i1 = ps::kOccGridMaxXY;

  int j0 = ps::kOccGridMinXY;
  int j1 = ps::kOccGridMaxXY;

  int k0 = ps::kOccGridMinZ;
  int k1 = ps::kOccGridMaxZ;

  for (int ii = i0; ii <= i1; ii++) {
    for (int jj = j0; jj <= j1; jj++) {
      for (int kk = k0; kk <= k1; kk++) {
        rt::Location loc(ii, jj, kk);

        float p = og.GetProbability(loc);
        save_file_.write(reinterpret_cast<const char*>(&p), sizeof(float));
      }
    }
  }
}

void FilterExtractor::WriteFilter(int frame) {
  // Get bounds
  int i0 = ps::kOccGridMinXY;
  int i1 = ps::kOccGridMaxXY;

  int j0 = ps::kOccGridMinXY;
  int j1 = ps::kOccGridMaxXY;

  for (int ii = i0; ii <= i1; ii++) {
    for (int jj = j0; jj <= j1; jj++) {
      double x1 = ii * ps::kResolution;
      double y1 = jj * ps::kResolution;
      double z1 = 0.0;

      Eigen::Vector2f pos1(x1, y1);

      // Get object type
      kt::ObjectClass c = kt::GetObjectTypeAtLocation(&tracklets_, pos1, frame, ps::kResolution);

      int label = -1;

      if (camera_cal_.InCameraView(x1, y1, z1) && c == kt::ObjectClass::NO_OBJECT) {
        label = 0;
      } else if (c != kt::ObjectClass::NO_OBJECT) {
        label = 1;
      }

      // Write it out
      save_file_.write(reinterpret_cast<const char*>(&label), sizeof(int));
    }
  }
}

void FilterExtractor::Run() {
  for (size_t scan_at = 0; scan_at < scans_.size(); scan_at++) {
    const auto &scan = scans_[scan_at];

    rt::OccGrid og = og_builder_.GenerateOccGrid(scan.GetHits());

    printf("\tWriting Occ Grid %ld\n", scan_at);
    WriteOccGrid(og);

    printf("\tWriting Filter %ld\n", scan_at);
    WriteFilter(scan_at);

    printf("\n");
  }
}

} // data
} // app
