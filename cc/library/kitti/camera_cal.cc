#include "library/kitti/camera_cal.h"

#include <Eigen/Dense>
#include <Eigen/LU>

namespace library {
namespace kitti {

CameraCal::CameraCal(const std::string &dirname) {
  FILE *f_cc = fopen( (dirname + "/calib_cam_to_cam.txt").c_str(), "r");
  FILE *f_vc = fopen( (dirname + "/calib_velo_to_cam.txt").c_str(), "r");

  LoadIntrinsics(f_cc);
  LoadExtrinsics(f_vc);

  fclose(f_cc);
  fclose(f_vc);

  std::cout << "P_rect: " << std::endl << P_rect_ << std::endl;
  std::cout << "R_rect: " << std::endl << R_rect_ << std::endl;
  std::cout << "T_cv: " << std::endl << T_cv_ << std::endl;

  M_ = P_rect_ * R_rect_ * T_cv_;
}

void CameraCal::LoadIntrinsics(FILE *f_cc) {
  int cam = 2;

  char p_str[100];
  sprintf(p_str, "P_rect_%02d: ", cam);

  char r_str[100];
  //sprintf(r_str, "R_rect_%02d: ", cam);
  sprintf(r_str, "R_rect_%02d: ", 0); // according to kitti paper

  char *line = NULL;
  size_t len;

  double p[12];
  double r[9];

  while (getline(&line, &len, f_cc) != -1) {
    if (strncmp(r_str, line, strlen(r_str)) == 0) {
      sscanf(&line[strlen(r_str)], "%lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
              &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7], &r[8]);
    } else if (strncmp(p_str, line, strlen(p_str)) == 0) {
      sscanf(&line[strlen(p_str)], "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
              &p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6], &p[7], &p[8], &p[9], &p[10], &p[11]);
    }
  }

  R_rect_ = Eigen::MatrixXd(4, 4);
  R_rect_.setZero();
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      R_rect_(i, j) = r[i*3 + j];
    }
  }
  R_rect_(3, 3) = 1.0;

  P_rect_ = Eigen::MatrixXd(3, 4);
  for (int i=0; i<3; i++) {
    for (int j=0; j<4; j++) {
      P_rect_(i, j) = p[i*4 + j];
    }
  }
}

void CameraCal::LoadExtrinsics(FILE *f_vc) {
  char calib_date[100], calib_time[100];
  double R[9];
  double t[3];
  fscanf(f_vc, "calib_time: %s %s\nR: %lf %lf %lf %lf %lf %lf %lf %lf %lf\nT: %lf %lf %lf", calib_date, calib_time,
          &R[0], &R[1], &R[2], &R[3], &R[4], &R[5],
          &R[6], &R[7], &R[8], &t[0], &t[1], &t[2]);

  //printf("Read: %s %s\nR: %f %f %f %f %f %f %f %f %f\nT: %f %f %f\n", calib_date, calib_time,
  //        R[0], R[1], R[2], R[3], R[4], R[5],
  //        R[6], R[7], R[8], t[0], t[1], t[2]);

  T_cv_ = Eigen::MatrixXd(4, 4);
  T_cv_.setZero();
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      T_cv_(i, j) = R[i*3 + j];
    }
    T_cv_(i, 3) = t[i];
  }
  T_cv_(3, 3) = 1;
}

bool CameraCal::InCameraView(double x, double y, double z) const {
  Eigen::MatrixXd p_x(4, 1);
  p_x(0) = x;
  p_x(1) = y;
  p_x(2) = z;
  p_x(3) = 1.0;

  Eigen::MatrixXd p_c = P_rect_ * R_rect_ * T_cv_ * p_x;

  if (p_c(2) < 0) {
    return false;
  }

  double x_c = p_c(0) / p_c(2);
  double y_c = p_c(1) / p_c(2);

  if (x_c < 0 || x_c > 1392) {
    return false;
  }

  if (y_c < 0 || y_c > 512) {
    return false;
  }

  return true;
}

} // namespace kitti
} // namespace library
