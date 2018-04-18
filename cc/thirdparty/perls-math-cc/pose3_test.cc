#include "pose3.h"

#include <gtest/gtest.h>

#include "test_util.h"

namespace perls_math_cc {
namespace {

// test fixture
template <class Scalar>
class Pose3Test : public ::testing::Test {
  // This 'protected:' is optional.  There's no harm in making all
  // members of this fixture class template public.
 public:
  // Jacobian types
  using JminusType = ::Eigen::Matrix<Scalar, 6, 6>;
  using JplusType  = ::Eigen::Matrix<Scalar, 6, 12>;
 protected:
  std::vector<std::initializer_list<Scalar>> pose_data_;
  std::vector<Pose3<Scalar>> poses_;
  std::vector<::Sophus::Vector6<Scalar>> representations_;
  std::vector<::Sophus::Matrix3<Scalar>> rotation_matrices_;
  std::vector<::Sophus::Vector3<Scalar>> translations_;
  std::vector<Pose3<Scalar>> inverses_;
  std::vector<JminusType> inverse_jacobians_;
  std::vector<Pose3<Scalar>> head2tails_;
  std::vector<JplusType> head2tail_jacobians_;
  std::vector<Pose3<Scalar>> tail2tails_;
  std::vector<JplusType> tail2tail_jacobians_;
  const int kN_ = 3; // number of poses
  Pose3Test() {
    // initialize pose data
    pose_data_.push_back(
      { -2.931295, -1.068366,  1.197799, -0.487031,  1.142353, -1.569989 });
    pose_data_.push_back(
      { -1.246234, -2.984991, -0.848276,  1.667438,  0.818924, -2.288786 });
    pose_data_.push_back(
      { -2.628431,  0.320167,  0.406220,  1.528514,  0.058603, -1.766519 });
    // initialize poses
    for (auto pose_data_it = pose_data_.begin();
         pose_data_it != pose_data_.end();
         pose_data_it++) {
      poses_.push_back(*pose_data_it);
    }
    // initialize representations
    for (auto pose_data_it = pose_data_.begin();
         pose_data_it != pose_data_.end();
         pose_data_it++) {
      ::Sophus::Vector6<Scalar> representation;
      auto it = (*pose_data_it).begin();
      representation << *it, *(it+1), *(it+2), *(it+3), *(it+4), *(it+5);
      representations_.push_back(representation);
    }
    // initialize rotation matrices
    ::Sophus::Matrix3<Scalar> R;
    R <<  0.000335,  0.883383,  0.468652,
         -0.415455,  0.426416, -0.803472,
         -0.909614, -0.194434,  0.367149;
    rotation_matrices_.push_back(R);
    R << -0.449332, -0.550945, -0.703250,
         -0.514393, -0.484049,  0.707881,
         -0.730411,  0.679821, -0.065904;
    rotation_matrices_.push_back(R);
    R << -0.194141,  0.030083, -0.980512,
         -0.979224, -0.065621,  0.191873,
         -0.058570,  0.997391,  0.042198;
    rotation_matrices_.push_back(R);
    // initialize translations
    for (auto representations_it = representations_.begin();
         representations_it != representations_.end();
         representations_it++) {
      translations_.push_back((*representations_it).head(3));
    }
    // initialize inverses
    inverses_.push_back(Pose3<Scalar>(
      {  0.646660,  3.277917,  0.075586, -1.142176, -0.487765,  1.570417 }));
    inverses_.push_back(Pose3<Scalar>(
      { -2.715022, -1.554813,  1.180701,  1.663629,  0.779959, -2.254956 }));
    inverses_.push_back(Pose3<Scalar>(
      { -0.172980, -0.305080, -2.655781,  1.354318,  1.373052,  2.987862 }));
    // initialize inverse jacobians
    JminusType Jminus;
    Jminus <<
        -0.000335,  0.415455,  0.909614,  0.000000,  1.467280,  1.218179,
        -0.883383, -0.426416,  0.194434,  0.075586, -0.302639, -0.306174,
        -0.468652,  0.803472, -0.367149, -3.277917,  0.571470,  2.855905,
         0.000000,  0.000000,  0.000000, -0.000430, -1.000389,  0.220493,
         0.000000,  0.000000,  0.000000,  1.000000, -0.000335, -0.909540,
         0.000000,  0.000000,  0.000000,  0.000201,  0.000831, -0.470483;
    inverse_jacobians_.push_back(Jminus);
    Jminus <<
         0.449332,  0.514393,  0.730411,  0.000000,  1.661485, -0.700196,
         0.550945,  0.484049, -0.679821,  1.180701, -2.702353, -1.041328,
         0.703250, -0.707881,  0.065904,  1.554813,  0.261976, -2.981381,
         0.000000,  0.000000,  0.000000,  0.888991, -0.105178,  0.091697,
         0.000000,  0.000000,  0.000000, -0.774950, -0.060985,  0.995694,
         0.000000,  0.000000,  0.000000,  0.625183,  0.921367,  0.130390;
    inverse_jacobians_.push_back(Jminus);
    Jminus <<
         0.194141,  0.979224,  0.058570,  0.000000,  0.417067,  2.635979,
        -0.030083,  0.065621, -0.997391, -2.655781, -0.172825,  0.162848,
         0.980512, -0.191873, -0.042198,  0.305080, -0.007312, -0.190397,
         0.000000,  0.000000,  0.000000,  5.030117, -0.032947, -1.072012,
         0.000000,  0.000000,  0.000000,  0.153126,  0.041772,  0.976661,
         0.000000,  0.000000,  0.000000,  4.932091,  0.966803, -1.093318;
    inverse_jacobians_.push_back(Jminus);
    // initialize head2tails ( head2tail(poses[i], poses[(i+1)%3]) )
    head2tails_.push_back(Pose3<Scalar>(
      { -5.966148, -1.141894,  2.600333,  1.056054, -0.242947,  2.533913 }));
    head2tails_.push_back(Pose3<Scalar>(
      { -0.527266, -1.500365,  1.262444, -0.155536,  0.546890,  0.672967 }));
    head2tails_.push_back(Pose3<Scalar>(
      { -3.265941,  3.490492, -0.437130,  2.719393,  0.469874, -0.166302 }));
    // initialize head2tail jacobians
    JplusType Jplus;
    Jplus <<
         1.000000,  0.000000,  0.000000, -0.649571,  0.001132,  0.073527, \
         0.000335,  0.883383,  0.468652,  0.000000,  0.000000,  0.000000,
         0.000000,  1.000000,  0.000000,  2.760074, -1.402533, -3.034853, \
        -0.415455,  0.426416, -0.803472,  0.000000,  0.000000,  0.000000,
         0.000000,  0.000000,  1.000000, -1.260869, -0.071078,  0.000000, \
        -0.909614, -0.194434,  0.367149,  0.000000,  0.000000,  0.000000,
         0.000000,  0.000000,  0.000000, -0.244670, -0.845339,  0.000000, \
         0.000000,  0.000000,  0.000000,  1.000000,  0.142262, -0.869026,
         0.000000,  0.000000,  0.000000,  0.340887, -0.571626,  0.000000, \
         0.000000,  0.000000,  0.000000,  0.000000,  0.818855,  0.392047,
         0.000000,  0.000000,  0.000000, -0.850755,  0.203359,  1.000000, \
         0.000000,  0.000000,  0.000000,  0.000000, -0.591367,  0.576205;
    head2tail_jacobians_.push_back(Jplus);
    Jplus <<
         1.000000,  0.000000,  0.000000, -0.001353, -1.388583, -1.484626, \
        -0.449332, -0.550945, -0.703250,  0.000000,  0.000000,  0.000000,
         0.000000,  1.000000,  0.000000,  0.423271, -1.589646,  0.718968, \
        -0.514393, -0.484049,  0.707881,  0.000000,  0.000000,  0.000000,
         0.000000,  0.000000,  1.000000, -0.297257,  1.591105,  0.000000, \
        -0.730411,  0.679821, -0.065904,  0.000000,  0.000000,  0.000000,
         0.000000,  0.000000,  0.000000, -0.786742,  0.209416,  0.000000, \
         0.000000,  0.000000,  0.000000,  1.000000, -0.604934, -0.127257,
         0.000000,  0.000000,  0.000000, -0.122171, -0.983872,  0.000000, \
         0.000000,  0.000000,  0.000000,  0.000000, -0.113011,  0.991888,
         0.000000,  0.000000,  0.000000, -1.139543,  0.108903,  1.000000, \
         0.000000,  0.000000,  0.000000,  0.000000, -1.163260, -0.132082;
    head2tail_jacobians_.push_back(Jplus);
    Jplus <<
         1.000000,  0.000000,  0.000000,  1.011513,  0.164011, -3.170325, \
        -0.194141,  0.030083, -0.980512,  0.000000,  0.000000,  0.000000,
         0.000000,  1.000000,  0.000000, -0.126390,  0.827248, -0.637511, \
        -0.979224, -0.065621,  0.191873,  0.000000,  0.000000,  0.000000,
         0.000000,  0.000000,  1.000000, -1.239757,  2.985815,  0.000000, \
        -0.058570,  0.997391,  0.042198,  0.000000,  0.000000,  0.000000,
         0.000000,  0.000000,  0.000000, -0.032935,  1.121062,  0.000000, \
         0.000000,  0.000000,  0.000000,  1.000000, -0.032899, -1.120142,
         0.000000,  0.000000,  0.000000, -0.997851, -0.029417,  0.000000, \
         0.000000,  0.000000,  0.000000,  0.000000, -0.997899,  0.026916,
         0.000000,  0.000000,  0.000000, -0.073482,  0.507588,  1.000000, \
         0.000000,  0.000000,  0.000000,  0.000000, -0.072661, -0.464974;
    head2tail_jacobians_.push_back(Jplus);
    // initialize tail2tails ( tail2tail(poses[i], poses[(i+1)%3]) )
    tail2tails_.push_back(Pose3<Scalar>(
      {  2.657974,  1.069102,  1.578447,  2.750574,  0.065496, -0.495281 }));
    tail2tails_.push_back(Pose3<Scalar>(
      { -1.995385,  0.014488,  3.229013, -0.160700,  0.585702,  0.706757 }));
    tail2tails_.push_back(Pose3<Scalar>(
      {  1.372121,  0.871520,  0.063942, -2.217339,  0.118705, -1.089087 }));
    // initialize tail2tail jacobians
    Jplus <<
        -0.000335,  0.415455,  0.909614,  0.000000, -0.894572,  0.699424, \
         0.000335, -0.415455, -0.909614,  0.000000,  0.000000,  0.000000,
        -0.883383, -0.426416,  0.194434,  1.578447, -1.243942, -2.411649, \
         0.883383,  0.426416, -0.194434,  0.000000,  0.000000,  0.000000,
        -0.468652,  0.803472, -0.367149, -1.069102,  2.348922,  0.455668, \
         0.468652, -0.803472,  0.367149,  0.000000,  0.000000,  0.000000,
         0.000000,  0.000000,  0.000000, -0.881726,  0.420919,  0.709421, \
         0.000000,  0.000000,  0.000000,  1.000000,  0.057944, -0.709421,
         0.000000,  0.000000,  0.000000, -0.475279, -0.777534,  0.603391, \
         0.000000,  0.000000,  0.000000,  0.000000,  0.468560, -0.603391,
         0.000000,  0.000000,  0.000000, -0.057708, -0.440455, -0.320718, \
         0.000000,  0.000000,  0.000000,  0.000000,  0.885330,  0.320718;
    tail2tail_jacobians_.push_back(Jplus);
    Jplus <<
         0.449332,  0.514393,  0.730411,  0.000000,  0.297151, -2.196104, \
        -0.449331, -0.514393, -0.730411,  0.000000,  0.000000,  0.000000,
         0.550945,  0.484049, -0.679821,  3.229013, -1.986074, -2.490012, \
        -0.550945, -0.484049,  0.679821,  0.000000,  0.000000,  0.000000,
         0.703250, -0.707881,  0.065904, -0.014488,  0.192537, -1.345921, \
        -0.703250,  0.707881, -0.065904,  0.000000,  0.000000,  0.000000,
         0.000000,  0.000000,  0.000000, -0.912576,  0.075191,  0.136804, \
         0.000000,  0.000000,  0.000000,  1.000000, -0.658703, -0.136804,
         0.000000,  0.000000,  0.000000,  0.649371,  0.073379, -0.991292, \
         0.000000,  0.000000,  0.000000,  0.000000, -0.118141,  0.991292,
         0.000000,  0.000000,  0.000000, -0.504458,  1.036898,  0.141527, \
         0.000000,  0.000000,  0.000000,  0.000000, -1.191609, -0.141527;
    tail2tail_jacobians_.push_back(Jplus);
    Jplus <<
         0.194141,  0.979224,  0.058570, -0.000000, -0.873446, -0.027002, \
        -0.194141, -0.979223, -0.058570,  0.000000,  0.000000,  0.000000,
        -0.030083,  0.065621, -0.997391,  0.063947,  1.370897, -0.061646, \
         0.030083, -0.065621,  0.997391,  0.000000,  0.000000,  0.000000,
         0.980512, -0.191873, -0.042198, -0.871522,  0.058000,  1.419587, \
        -0.980512,  0.191873,  0.042198,  0.000000,  0.000000,  0.000000,
         0.000000,  0.000000,  0.000000, -0.466578,  0.037725,  0.917485, \
         0.000000,  0.000000,  0.000000,  1.000000, -0.117752, -0.917484,
         0.000000,  0.000000,  0.000000, -0.886206, -0.019583, -0.410181, \
         0.000000,  0.000000,  0.000000,  0.000000, -0.158837,  0.410181,
         0.000000,  0.000000,  0.000000, -0.055255,  1.003576,  0.066457, \
         0.000000,  0.000000,  0.000000,  0.000000, -0.994302, -0.066457;
    tail2tail_jacobians_.push_back(Jplus);
  }
  virtual ~Pose3Test() = default;
  virtual void SetUp() { }
  virtual void TearDown() { }
};

typedef ::testing::Types<float, double> ScalarTypes;
TYPED_TEST_CASE(Pose3Test, ScalarTypes);

TYPED_TEST(Pose3Test, HasExpectedRotationTranslation) {
  for (int i = 0; i < this->kN_; i++) {
    EXPECT_MATRICES_APPROX_EQ(this->poses_[i].so3().matrix(),
                              this->rotation_matrices_[i]);
    EXPECT_MATRICES_APPROX_EQ(this->poses_[i].translation(),
                              this->translations_[i]);
  }
}

TYPED_TEST(Pose3Test, CanConvertToAndFromSE3) {
  for (int i = 0; i < this->kN_; i++) {
    // convert to SE3
    ::Sophus::SE3<TypeParam> pose_as_se3 =
        static_cast<::Sophus::SE3<TypeParam>>(this->poses_[i]);
    EXPECT_MATRICES_APPROX_EQ(this->poses_[i].matrix(),
                              pose_as_se3.matrix());
    // convert back to Pose3
    Pose3<TypeParam> pose_from_se3 =
        static_cast<Pose3<TypeParam>>(pose_as_se3);
    EXPECT_MATRICES_APPROX_EQ(this->poses_[i].matrix(),
                              pose_from_se3.matrix());
  }
}

TYPED_TEST(Pose3Test, PoseRepresentation) {
  for (int i = 0; i < this->kN_; i++) {
    EXPECT_MATRICES_APPROX_EQ(this->poses_[i].xyzrph(),
                              this->representations_[i]);
  }
}

TYPED_TEST(Pose3Test, HeadToTail) {
  ::Eigen::Matrix<TypeParam, 6, 12> J;
  for (int i = 0; i < this->kN_; i++) {
    Pose3<TypeParam> head2tail =
        this->poses_[i].HeadToTail(this->poses_[(i + 1) % this->kN_], &J);
    EXPECT_MATRICES_APPROX_EQ(head2tail.xyzrph(),
                              this->head2tails_[i].xyzrph());
    EXPECT_MATRICES_APPROX_EQ(J, this->head2tail_jacobians_[i]);
  }
}

TYPED_TEST(Pose3Test, TailToTail) {
  ::Eigen::Matrix<TypeParam, 6, 12> J;
  for (int i = 0; i < this->kN_; i++) {
    Pose3<TypeParam> tail2tail =
        this->poses_[i].TailToTail(this->poses_[(i + 1) % this->kN_], &J);
    EXPECT_MATRICES_APPROX_EQ(tail2tail.xyzrph(),
                              this->tail2tails_[i].xyzrph());
    EXPECT_MATRICES_APPROX_EQ(J, this->tail2tail_jacobians_[i]);
  }
}

TYPED_TEST(Pose3Test, Inverse) {
  ::Eigen::Matrix<TypeParam, 6, 6> J;
  for (int i = 0; i < this->kN_; i++) {
    Pose3<TypeParam> inverse = this->poses_[i].Inverse(&J);
    EXPECT_MATRICES_APPROX_EQ(inverse.xyzrph(),
                              this->inverses_[i].xyzrph());
    EXPECT_MATRICES_APPROX_EQ(J, this->inverse_jacobians_[i]);
  }
}

} // namespace
} // namespace perls_math_cc
