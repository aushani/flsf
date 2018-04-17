#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace library {
namespace kitti {

class Pose {
 public:
  double x, y, z, r, p, h;
  double v_f, v_h;

  static std::vector<Pose> LoadRawPoses(const fs::path &path);
  static std::vector<Pose> LoadScanMatchedPoses(const fs::path &path);

 private:
  Pose();

  struct oxt_t {
    float lat;
    float lon;
    float alt;
    float roll;
    float pitch;
    float yaw;
    float vn;
    float ve;
    float vf;
    float vl;
    float vu;
    float ax;
    float ay;
    float az;
    float af;
    float al;
    float au;
    float wx;
    float wy;
    float wz;
    float wf;
    float wl;
    float wu;
    float posacc;
    float velacc;
    int navstat;
    int numsats;
    int posmode;
    int velmode;
    int orimode;
  };

  static oxt_t ReadOxtFile(FILE *fp);
  static void LatLonToMercator(double lat, double lon, double scale, double *x, double *y);
};

} // kitti
} // library
