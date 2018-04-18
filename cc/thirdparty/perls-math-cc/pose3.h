// Copyright 2017 Derrick Dominic

#ifndef PERLSMATHCC_SRC_POSE3_H_
#define PERLSMATHCC_SRC_POSE3_H_

#include <iomanip> // std::setprecision
#include <initializer_list>

#include <Eigen/Geometry>
#include "thirdparty/sophus/se3.hpp"
#include "thirdparty/sophus/types.hpp"

// follows the same structure as sophus/se3.hpp

namespace thirdparty {
namespace perls_math_cc {

// shorthand for double and float templated Pose3
template <class Scalar_, int Options = 0>
class Pose3;
using Pose3d = Pose3<double>;
using Pose3f = Pose3<float>;

// Implements Pose3 class.
// Pose3 is the SSC representaiton of SE(3).
template <class Scalar_, int Options>
class Pose3 : public thirdparty::Sophus::SE3<Scalar_, Options> {
 public:
  using Scalar      = Scalar_;
  using SE3Type     = thirdparty::Sophus::SE3<Scalar_, Options>;
  using SO3Type     = thirdparty::Sophus::SO3<Scalar_>;
  using Vector3Type = thirdparty::Sophus::Vector3<Scalar_>;
  using Vector6Type = thirdparty::Sophus::Vector6<Scalar_>;
  using Matrix3Type = thirdparty::Sophus::Matrix3<Scalar_>;
  using Matrix4Type = thirdparty::Sophus::Matrix4<Scalar_>;

  // Default constructor
  Pose3() : SE3Type() { }

  // Copy constructor
  template <class OtherScalar>
  Pose3(const Pose3<OtherScalar> &other) {
    this->so3_ = other.so3();
    this->translation_ = other.translation();
  }

  void Set(Vector6Type pose_vector) {
    Scalar x = pose_vector[0], y = pose_vector[1], z = pose_vector[2],
           r = pose_vector[3], p = pose_vector[4], h = pose_vector[5];

    // construct rotation matrix
    Scalar sr, sp, sh, cr, cp, ch;
    sr = ::std::sin(r);
    sp = ::std::sin(p);
    sh = ::std::sin(h);
    cr = ::std::cos(r);
    cp = ::std::cos(p);
    ch = ::std::cos(h);

    Matrix3Type R;
    R <<  ch*cp, -sh*cr + ch*sp*sr,  sh*sr + ch*sp*cr,
          sh*cp,  ch*cr + sh*sp*sr, -ch*sr + sh*sp*cr,
         -sp,     cp*sr,             cp*cr;

    this->so3_ = SO3Type(R);
    this->translation_ = Vector3Type(x, y, z);
  }

  // Construct from initializer list
  Pose3(std::initializer_list<Scalar> list) {
    Vector6Type pose_vector;
    auto it = list.begin();
    pose_vector << *it, *(it+1), *(it+2), *(it+3), *(it+4), *(it+5);
    Set(pose_vector);
  }

  // Construct from pose representatation (x, y, theta, roll, pitch, heading)
  Pose3(Vector6Type pose_vector) {
    Set(pose_vector);
  }

  // Constructor to convert back from SE3
  // Required for operator overloading with assignment to work for Pose3
  Pose3(const SE3Type &pose_se3) : SE3Type(pose_se3) { }

  // compute and return pose representation as Eigen::Vector6
  // lowercase to be consistent with so3(), translation(), data()
  Vector6Type xyzrph() const {
    Matrix3Type R = this->rotationMatrix();
    Scalar h  = ::std::atan2(R(1, 0), R(0, 0));
    Scalar sh = ::std::sin(h),
           ch = ::std::cos(h);
    Scalar p  = ::std::atan2(-R(2, 0), (R(0, 0) * ch) + (R(1 ,0) * sh));
    Scalar r  = ::std::atan2((R(0, 2) * sh) - (R(1, 2) * ch),
                            -(R(0, 1) * sh) + (R(1, 1) * ch));

    Vector6Type pose_vector;
    pose_vector <<
        this->translation_[0],
        this->translation_[1],
        this->translation_[2],
        r, p, h;

    return pose_vector;
  }

  // ssc methods are mixed case
  Pose3<Scalar> HeadToTail(const Pose3<Scalar> &other,
                           ::Eigen::Matrix<Scalar, 6, 12> *J = nullptr) {
    Pose3<Scalar> head2tail = (*this) * other;

    if (J) {
      Vector6Type X_ij = this->xyzrph(),
                  X_jk = other.xyzrph(),
                  X_ik = head2tail.xyzrph();

      Matrix3Type R_ij = this->so3().matrix(),
                  R_jk = other.so3().matrix();

      const Scalar x_ij = X_ij[0], y_ij = X_ij[1], z_ij = X_ij[2],
                   r_ij = X_ij[3], p_ij = X_ij[4], h_ij = X_ij[5];

      const Scalar x_jk = X_jk[0], y_jk = X_jk[1], z_jk = X_jk[2],
                   r_jk = X_jk[3], p_jk = X_jk[4], h_jk = X_jk[5];

      const Scalar x_ik = X_ik[0], y_ik = X_ik[1], z_ik = X_ik[2],
                   r_ik = X_ik[3], p_ik = X_ik[4], h_ik = X_ik[5];

      Scalar sr_ij = ::std::sin(r_ij),
             cr_ij = ::std::cos(r_ij),
             sp_ij = ::std::sin(p_ij),
             cp_ij = ::std::cos(p_ij),
             sh_ij = ::std::sin(h_ij),
             ch_ij = ::std::cos(h_ij);

      Scalar sp_jk = ::std::sin(p_jk),
             cp_jk = ::std::cos(p_jk);

      Scalar sr_ik = ::std::sin(r_ik),
             cr_ik = ::std::cos(r_ik),
             sp_ik = ::std::sin(p_ik),
             cp_ik = ::std::cos(p_ik),
             tp_ik = ::std::tan(p_ik),
             sh_ik = ::std::sin(h_ik),
             ch_ik = ::std::cos(h_ik);

      Scalar sh_dif = ::std::sin(h_ik - h_ij),
             ch_dif = ::std::cos(h_ik - h_ij),
             sr_dif = ::std::sin(r_ik - r_jk),
             cr_dif = ::std::cos(r_ik - r_jk);

      Scalar x_dif = x_ik - x_ij, y_dif = y_ik - y_ij, z_dif = z_ik - z_ij;

      Scalar secp_ik = 1./cp_ik;

      // M
      ::Eigen::Matrix<Scalar, 3, 3> M;
      M << R_ij(0,2)*y_jk - R_ij(0,1)*z_jk, z_dif*ch_ij, -y_dif,
           R_ij(1,2)*y_jk - R_ij(1,1)*z_jk, z_dif*sh_ij,  x_dif,
           R_ij(2,2)*y_jk - R_ij(2,1)*z_jk,
           -x_jk*cp_ij - (y_jk*sr_ij + z_jk*cr_ij)*sp_ij,  0;

      // K1
      ::Eigen::Matrix<Scalar, 3, 3> K1;
      K1 <<                        cp_ij*ch_dif*secp_ik, sh_dif*secp_ik, 0,
                                          -cp_ij*sh_dif,         ch_dif, 0,
            (R_jk(0,1)*sr_ik + R_jk(0,2)*cr_ik)*secp_ik,   sh_dif*tp_ik, 1;

      ::Eigen::Matrix<Scalar, 3, 3> K2;
      K2 << 1,   sr_dif*tp_ik, (R_ij(0,2)*ch_ik + R_ij(1,2)*sh_ik)*secp_ik,
            0,         cr_dif,                               -cp_jk*sr_dif,
            0, sr_dif*secp_ik,                        cp_jk*cr_dif*secp_ik;

      // Jplus
      *J = ::Eigen::Matrix<Scalar, 6, 12>::Zero();
      J->block(0, 0, 3, 3) = ::Eigen::Matrix<Scalar, 3, 3>::Identity();
      J->block(0, 3, 3, 3) = M;
      J->block(0, 6, 3, 3) = R_ij;
      J->block(3, 3, 3, 3) = K1;
      J->block(3, 9, 3, 3) = K2;
    }

    return head2tail;
  }

  // ssc methods are mixed case
  Pose3<Scalar> TailToTail(const Pose3<Scalar> &other,
                           ::Eigen::Matrix<Scalar, 6, 12> *J = nullptr) {
    if (J) {
      ::Eigen::Matrix<Scalar, 6, 6> Jminus;
      Pose3<Scalar> inverse = this->Inverse(&Jminus);

      ::Eigen::Matrix<Scalar, 6, 12> Jplus;
      Pose3<Scalar> head2tail = inverse.HeadToTail(other, &Jplus);

      J->block(0, 0, 6, 6) = Jplus.block(0, 0, 6, 6) * Jminus;
      J->block(0, 6, 6, 6) = Jplus.block(0, 6, 6, 6);

      return head2tail;
    }

    else {
      return this->inverse() * other;
    }
  }

  // ssc methods are mixed case (distinguish from inverse)
  // overriding inverse so the output is cast to Pose3 instead of SE3
  // also provides analytical Jacobian
  Pose3<Scalar> Inverse(::Eigen::Matrix<Scalar, 6, 6> *J = nullptr) {
    Pose3<Scalar> inverse = this->inverse();

    if (J) {
      Vector6Type X_ij = this->xyzrph(), X_ji = inverse.xyzrph();

      const Scalar x_ij = X_ij[0], y_ij = X_ij[1], z_ij = X_ij[2],
                   r_ij = X_ij[3], p_ij = X_ij[4], h_ij = X_ij[5];

      const Scalar y_ji = X_ji[1], z_ji = X_ji[2];

      Scalar sr_ij = ::std::sin(r_ij),
             cr_ij = ::std::cos(r_ij),
             sp_ij = ::std::sin(p_ij),
             cp_ij = ::std::cos(p_ij),
             sh_ij = ::std::sin(h_ij),
             ch_ij = ::std::cos(h_ij);

      Matrix3Type R_ij = this->so3().matrix();

      // N
      double tmp = x_ij*ch_ij + y_ij*sh_ij;
      ::Eigen::Matrix<Scalar, 3, 3> N;
      N <<
          0,       -R_ij(2,0)*tmp + z_ij*cp_ij, R_ij(1,0)*x_ij - R_ij(0,0)*y_ij,
       z_ji, -R_ij(2,1)*tmp + z_ij*sp_ij*sr_ij, R_ij(1,1)*x_ij - R_ij(0,1)*y_ij,
      -y_ji, -R_ij(2,2)*tmp + z_ij*sp_ij*cr_ij, R_ij(1,2)*x_ij - R_ij(0,2)*y_ij;

      // Q
      tmp = 1-R_ij(0,2)*R_ij(0,2);
      double sqrt_tmp = ::std::sqrt(tmp);
      ::Eigen::Matrix<Scalar, 3, 3> Q;
      Q <<           -R_ij(0,0),          -R_ij(0,1)*cr_ij, R_ij(0,2)*R_ij(2,2),
             R_ij(0,1)*sqrt_tmp, -R_ij(2,2)*ch_ij*sqrt_tmp,  R_ij(1,2)*sqrt_tmp,
            R_ij(0,2)*R_ij(0,0),          -R_ij(1,2)*ch_ij,          -R_ij(2,2);

      Q *= 1./tmp;

      // Jminus
      *J = ::Eigen::Matrix<Scalar, 6, 6>::Zero();
      J->topLeftCorner(3, 3)     = -R_ij.transpose();
      J->topRightCorner(3, 3)    = N;
      J->bottomRightCorner(3, 3) = Q;
    }

    return inverse;
  }
};

// output stream
template <class Scalar>
std::ostream& operator<<(std::ostream& stream, const Pose3<Scalar> pose3) {
  stream << pose3.xyzrph();
  return stream;
}

}  // namespace perls_math_cc
}  // namespace thirdparty

#endif  // PERLSMATHCC_SRC_POSE3_H_
