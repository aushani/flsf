#include "library/kitti/pose.h"

#include <math.h>

#include <boost/format.hpp>

namespace library {
namespace kitti {

Pose::Pose() {
}

Pose::oxt_t Pose::ReadOxtFile(FILE *fp) {
  oxt_t oxt;

  int res =  fscanf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %d %d %d %d",
                      &oxt.lat,
                      &oxt.lon,
                      &oxt.alt,
                      &oxt.roll,
                      &oxt.pitch,
                      &oxt.yaw,
                      &oxt.vn,
                      &oxt.ve,
                      &oxt.vf,
                      &oxt.vl,
                      &oxt.vu,
                      &oxt.ax,
                      &oxt.ay,
                      &oxt.az,
                      &oxt.af,
                      &oxt.al,
                      &oxt.au,
                      &oxt.wx,
                      &oxt.wy,
                      &oxt.wz,
                      &oxt.wf,
                      &oxt.wl,
                      &oxt.wu,
                      &oxt.posacc,
                      &oxt.velacc,
                      &oxt.navstat,
                      &oxt.numsats,
                      &oxt.posmode,
                      &oxt.velmode,
                      &oxt.orimode);

  if (res!=30 && res!=26) {
    printf("Could not parse oxts file (only got %d)\n", res);
  }

  return oxt;
}

void Pose::LatLonToMercator(double lat, double lon, double scale, double *x, double *y) {
  // Adapted from KITTI dev kit
  // converts lat/lon coordinates to mercator coordinates using mercator scale

  double er = 6378137;
  *x = scale * lon * M_PI * er / 180;
  *y = scale * er * log( tan((90+lat) * M_PI / 360) );
}

std::vector<Pose> Pose::LoadRawPoses(const fs::path &path) {
  std::vector<Pose> poses;

  double scale = -1;

  int i = 0;
  while  (1) {
    // Load oxts
    std::string fn = (boost::format("%010d.txt") % i).str();
    fs::path filename = path / "oxts" / "data" / fn;

    if (!fs::exists(filename)) {
      break;
    }

    FILE *fp = fopen(filename.c_str(), "r");
    oxt_t oxt = ReadOxtFile(fp);

    if (i==0) {
      scale = cos(oxt.lat * M_PI / 180.0);
    }

    double pose[6];
    LatLonToMercator(oxt.lat, oxt.lon, scale, &pose[0], &pose[1]);
    pose[2] = oxt.alt;
    pose[3] = oxt.roll;
    pose[4] = oxt.pitch;
    pose[5] = oxt.yaw;

    Pose p;
    p.x = pose[0];
    p.y = pose[1];
    p.z = pose[2];
    p.r = pose[3];
    p.p = pose[4];
    p.h = pose[5];

    p.v_f = oxt.vf;
    p.v_h = oxt.wu;

    poses.push_back(p);

    fclose(fp);

    i++;
  }

  return poses;
}

std::vector<Pose> Pose::LoadScanMatchedPoses(const fs::path &path) {
  fs::path filename = path / "sm_poses.txt";
  BOOST_ASSERT(fs::exists(filename));

  FILE *fp = fopen(filename.c_str(), "r");

  std::vector<Pose> poses;

  while (true) {
    Pose p;
    int res = fscanf(fp, "%lf %lf %lf %lf %lf %lf", &p.x, &p.y, &p.z, &p.r, &p.p, &p.h);
    if (res!=6) {
      break;
    }

    poses.push_back(p);
  }

  fclose(fp);

  return poses;
}


} // namespace kitti
} // namespace library
