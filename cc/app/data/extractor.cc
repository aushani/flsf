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

void Extractor::Write(const rt::OccGrid &og, int i, int j) {
  // Get bounds
  int i0 = i - ps::kPatchSize / 2;
  int i1 = i0 + ps::kPatchSize;

  int j0 = j - ps::kPatchSize / 2;
  int j1 = j0 + ps::kPatchSize;

  int k0 = ps::kOccGridMinZ;
  int k1 = ps::kOccGridMaxZ;

  for (int ii = i0; ii < i1; ii++) {
    for (int jj = j0; jj < j1; jj++) {
      for (int kk = k0; kk < k1; kk++) {
        rt::Location loc(ii, jj, kk);

        float p = og.GetProbability(loc);
        save_file_ << p;
      }
    }
  }
}

void Extractor::ProcessOccGrids(const rt::OccGrid &og1, const rt::OccGrid &og2, int idx1, int idx2) {
  // Get vehicle poses
  kt::Pose p1 = sm_poses_[idx1];
  kt::Pose p2 = sm_poses_[idx2];

  double p = 1.0 / (ps::kSearchSize * ps::kSearchSize - 1);
  std::bernoulli_distribution rand(p);

  // Iterate through occ grid
  for (int i1=ps::kOccGridMinXY; i1<=ps::kOccGridMaxXY; i1++) {
    for (int j1=ps::kOccGridMinXY; j1<=ps::kOccGridMaxXY; j1++) {
      double x1 = i1 * ps::kResolution;
      double y1 = j1 * ps::kResolution;
      double z1 = 0.0;
      Eigen::Vector2f pos1(x1, y1);

      // Check to make sure we're in camera view
      if (!camera_cal_.InCameraView(x1, y1, z1)) {
        continue;
      }

      // Project position from og1 to og2
      Eigen::Vector2f pos2 = kt::FindCorrespondingPosition(&tracklets_, pos1, idx1, idx2, p1, p2);
      int i2_match = std::round(pos2.x() / ps::kResolution);
      int j2_match = std::round(pos2.y() / ps::kResolution);

      // Get object type
      kt::ObjectClass c = kt::GetObjectTypeAtLocation(&tracklets_, pos1, idx1, ps::kResolution);

      // Look through search spacej;w
      for (int di=ps::kMinSearchDist; di<=ps::kMaxSearchDist/2; di++) {
        for (int dj=ps::kMinSearchDist; dj<=ps::kMaxSearchDist/2; dj++) {
          int i2 = i1 + di;
          int j2 = j1 + dj;

          bool match = (i2 == i2_match) && (j2 == j2_match);

          // Need to do some kind of random filtering, otherwise way too much
          // data to store
          if (!match && !rand(random_generator_)) {
            continue;
          }

          // Now we can save this one!
          Write(og1, i1, j1);
          Write(og2, i2, j2);

          int object_type = kt::ObjectClassToInt(c);
          save_file_ << object_type;

          int match_flag = match ? 1:0;
          save_file_ << match_flag;

          count_written_++;
        }
      }
    }
  }
}

void Extractor::Run() {
  kt::VelodyneScan first = scans_[0];
  rt::OccGrid prev = og_builder_.GenerateOccGrid(first.GetHits());

  for (size_t scan_at = 1; scan_at < scans_.size(); scan_at++) {
    kt::VelodyneScan scan = scans_[scan_at];
    rt::OccGrid next = og_builder_.GenerateOccGrid(scan.GetHits());

    ProcessOccGrids(prev, next, scan_at - 1, scan_at);
    prev = next;

    printf("Processed frame %ld, written %ld so far\n", scan_at, count_written_);
  }
}

} // data
} // app
