#include "app/data/extractor.h"

#include "library/ray_tracing/occ_grid_location.h"

#include "library/kitti/util.h"

namespace app {
namespace data {

Extractor::Extractor(const fs::path &base_path, const fs::path &save_path) :
 scans_(kt::VelodyneScan::LoadDirectory(base_path / "velodyne_points" / "data")),
 sm_poses_(kt::Pose::LoadScanMatchedPoses(base_path)),
 camera_cal_(base_path.parent_path()),
 og_builder_(ps::kMaxVelodyneScanPoints, ps::kResolution, ps::kMaxRange),
 random_generator_(random_device_()),
 save_file_( (save_path / "matches.bin").string(), std::ios::binary) {
  tracklets_.loadFromFile( (base_path / "tracklet_labels.xml").string() );
}

void Extractor::WriteOccGrid(const rt::OccGrid &og) {
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

void Extractor::WriteFilter(int frame) {
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

void Extractor::WriteFlow(int frame1, int frame2) {
  // Get vehicle poses
  kt::Pose p1 = sm_poses_[frame1];
  kt::Pose p2 = sm_poses_[frame2];

  // Get bounds
  int i0 = ps::kOccGridMinXY;
  int i1 = ps::kOccGridMaxXY;

  int j0 = ps::kOccGridMinXY;
  int j1 = ps::kOccGridMaxXY;

  for (int ii = i0; ii <= i1; ii++) {
    for (int jj = j0; jj <= j1; jj++) {
      double x1 = ii * ps::kResolution;
      double y1 = jj * ps::kResolution;

      Eigen::Vector2f pos1(x1, y1);

      // Project position
      Eigen::Vector2f pos2 = kt::FindCorrespondingPosition(&tracklets_, pos1, frame1, frame2, p1, p2);
      //int i2_match = std::round(pos2.x() / ps::kResolution);
      //int j2_match = std::round(pos2.y() / ps::kResolution);

      // Compute flow
      Eigen::Vector2f flow = pos2 - pos1;
      float flow_i = flow.x();
      float flow_j = flow.y();

      // Write it out
      save_file_.write(reinterpret_cast<const char*>(&flow_i), sizeof(float));
      save_file_.write(reinterpret_cast<const char*>(&flow_j), sizeof(float));
    }
  }
}

void Extractor::ProcessOccGrids(const rt::OccGrid &og1, const rt::OccGrid &og2, int idx1, int idx2) {
  // Write out out grids
  printf("\tWriting Occ Grids...\n");
  WriteOccGrid(og1);
  WriteOccGrid(og2);

  // Write out object types
  printf("\tWriting Filter...\n");
  WriteFilter(idx1);

  // Write out ground truth flow
  printf("\tWriting Flow...\n");
  WriteFlow(idx1, idx2);
}

void Extractor::Run() {
  kt::VelodyneScan first = scans_[0];
  rt::OccGrid prev = og_builder_.GenerateOccGrid(first.GetHits());

  for (size_t scan_at = 1; scan_at < scans_.size(); scan_at++) {
    kt::VelodyneScan scan = scans_[scan_at];
    rt::OccGrid next = og_builder_.GenerateOccGrid(scan.GetHits());

    ProcessOccGrids(prev, next, scan_at - 1, scan_at);
    prev = next;

    printf("Processed frame %ld / %ld\n", scan_at, scans_.size());
  }
}

} // data
} // app
