// Copyright 2017 Derrick Dominic

#ifndef PERLSMATHCC_SRC_POSE2_H_
#define PERLSMATHCC_SRC_POSE2_H_

#include <iomanip>  // std::setprecision
#include <initializer_list>

#include "thirdparty/sophus/se2.hpp"
#include "thirdparty/sophus/types.hpp"
#include <Eigen/Geometry>

// follows the same structure as sophus/se2.hpp

namespace thirdparty {
namespace perls_math_cc {

// shorthand for double and float templated Pose2
template <class Scalar_, int Options = 0>
class Pose2;
using Pose2d = Pose2<double>;
using Pose2f = Pose2<float>;

// Implements Pose2 class.
// Pose2 is the SSC representaiton of SE(2).
template <class Scalar_, int Options>
class Pose2 : public thirdparty::Sophus::SE2<Scalar_, Options> {
 public:
  using Scalar         = Scalar_;
  using SE2Type        = thirdparty::Sophus::SE2<Scalar_, Options>;
  using SO2Type        = thirdparty::Sophus::SO2<Scalar_>;
  using Vector2Type    = thirdparty::Sophus::Vector2<Scalar_>;
  using Vector3Type    = thirdparty::Sophus::Vector3<Scalar_>;
  using Matrix2Type    = thirdparty::Sophus::Matrix2<Scalar_>;
  using Matrix3Type    = thirdparty::Sophus::Matrix3<Scalar_>;
  using Rotation2DType = ::Eigen::Rotation2D<Scalar_>;

  // Default constructor
  Pose2() : SE2Type() { }

  // Copy constructor
  template <class OtherScalar>
  Pose2(const Pose2<OtherScalar> &other) {
    this->so2_ = other.so2();
    this->translation_ = other.translation();
  }

  // Construct from pose representatation (x, y, theta)
  explicit Pose2(Vector3Type pose_vector) {
    Scalar x = pose_vector[0], y = pose_vector[1], theta = pose_vector[2];

    this->so2_ = SO2Type(Rotation2DType(theta).toRotationMatrix());
    this->translation_ = Vector2Type(x, y);
  }

  // Constructor to convert back from SE2
  // Required for operator overloading with assignment to work for Pose2
  explicit Pose2(const SE2Type &pose_se2) : SE2Type(pose_se2) { }

  // compute and return pose representation as Eigen::Vector3
  Vector3Type xyt() const {
    auto unit_complex = this->so2().unit_complex();
    Scalar theta = ::std::atan2(unit_complex[1], unit_complex[0]);
    return { this->translation_[0], this->translation_[1], theta };
  }

  // ssc methods are mixed case
  Pose2<Scalar> HeadToTail(const Pose2<Scalar> &other,
                           ::Eigen::Matrix<Scalar, 3, 6> *J = nullptr) {
    Pose2<Scalar> head2tail = static_cast<Pose2<Scalar>>((*this) * other);

    if (J) {
      Vector3Type X_ij = this->xyt(),
                  X_ik = head2tail.xyt();

      const Scalar x_ij = X_ij[0], y_ij = X_ij[1], t_ij = X_ij[2];
      const Scalar x_ik = X_ik[0], y_ik = X_ik[1];

      Scalar st_ij = ::std::sin(t_ij),
             ct_ij = ::std::cos(t_ij);

      *J <<
          1, 0, -(y_ik - y_ij), ct_ij, -st_ij, 0,
          0, 1,  (x_ik - x_ij), st_ij,  ct_ij, 0,
          0, 0,              1,     0,      0, 1;
    }

    return head2tail;
  }

  // ssc methods are mixed case
  Pose2<Scalar> TailToTail(const Pose2<Scalar> &other,
                           ::Eigen::Matrix<Scalar, 3, 6> *J = nullptr) {
    Pose2<Scalar> tail2tail =
        static_cast<Pose2<Scalar>>(this->inverse() * other);

    if (J) {
      Vector3Type X_ij = this->xyt(),
                  X_jk = tail2tail.xyt();

      const Scalar t_ij = X_ij[2];
      const Scalar x_jk = X_jk[0], y_jk = X_jk[1];

      Scalar st_ij = ::std::sin(t_ij),
             ct_ij = ::std::cos(t_ij);

      *J <<
          -ct_ij, -st_ij,  y_jk,  ct_ij, st_ij, 0,
           st_ij, -ct_ij, -x_jk, -st_ij, ct_ij, 0,
               0,      0,    -1,      0,     0, 1;
    }

    return tail2tail;
  }

  // ssc methods are mixed case (distinguish from inverse)
  // overriding inverse so the output is cast to Pose2 instead of SE2
  // also provides analytical Jacobian
  Pose2<Scalar> Inverse(::Eigen::Matrix<Scalar, 3, 3> *J = nullptr) {
    Pose2<Scalar> inverse = static_cast<Pose2<Scalar>>(this->inverse());

    if (J) {
      Vector3Type X_ij = this->xyt(),
                  X_ji = inverse.xyt();

      const Scalar t_ij = X_ij[2];
      const Scalar x_ji = X_ji[0], y_ji = X_ji[1];

      Scalar st_ij = ::std::sin(t_ij),
             ct_ij = ::std::cos(t_ij);

      *J <<
          -ct_ij, -st_ij,  y_ji,
           st_ij, -ct_ij, -x_ji,
               0,      0,    -1;
    }

    return inverse;
  }
};

// output stream
template <class Scalar>
std::ostream& operator<<(std::ostream& stream, const Pose2<Scalar> pose2) {
  stream << pose2.xyt();
  return stream;
}

}  // namespace perls_math_cc
}  // namespace thirdparty

#endif  // PERLSMATHCC_SRC_POSE2_H_
