#include "pose2.h"

#include <gtest/gtest.h>

#include "test_util.h"

namespace perls_math_cc {
namespace {

// test fixture
template <class Scalar>
class Pose2Test : public ::testing::Test {
  // This 'protected:' is optional.  There's no harm in making all
  // members of this fixture class template public.
 public:
  // Jacobian types
  using JminusType = ::Eigen::Matrix<Scalar, 3, 3>;
  using JplusType  = ::Eigen::Matrix<Scalar, 3, 6>;
 protected:
  std::vector<::Sophus::Vector3<Scalar>> representations_;
  std::vector<Pose2<Scalar>> poses_;
  std::vector<::Sophus::Matrix2<Scalar>> rotation_matrices_;
  std::vector<::Sophus::Vector2<Scalar>> translations_;
  std::vector<Pose2<Scalar>> inverses_;
  std::vector<JminusType> inverse_jacobians_;
  std::vector<Pose2<Scalar>> head2tails_;
  std::vector<JplusType> head2tail_jacobians_;
  std::vector<Pose2<Scalar>> tail2tails_;
  std::vector<JplusType> tail2tail_jacobians_;
  const int kN_ = 3; // number of poses
  Pose2Test() {
    // initialize representations
    representations_.push_back(
      { -2.931295, -1.068366, -1.569989 });
    representations_.push_back(
      { -1.246234, -2.984991, -2.288786 });
    representations_.push_back(
      { -2.628431,  0.320167, -1.766519 });
    // initialize poses
    for (auto representations_it = representations_.begin();
         representations_it != representations_.end();
         representations_it++) {
      poses_.push_back(Pose2<Scalar>(*representations_it));
    }
    // initialize rotation matrices
    ::Sophus::Matrix2<Scalar> R;
    Scalar t = representations_[0][2];
    R << ::std::cos(t), -::std::sin(t),
         ::std::sin(t),  ::std::cos(t);
    rotation_matrices_.push_back(R);
    t = representations_[1][2];
    R << ::std::cos(t), -::std::sin(t),
         ::std::sin(t),  ::std::cos(t);
    rotation_matrices_.push_back(R);
    t = representations_[2][2];
    R << ::std::cos(t), -::std::sin(t),
         ::std::sin(t),  ::std::cos(t);
    rotation_matrices_.push_back(R);
    // initialize translations
    for (auto representations_it = representations_.begin();
         representations_it != representations_.end();
         representations_it++) {
      translations_.push_back((*representations_it).head(2));
    }
    // initialize inverses
    inverses_.push_back(Pose2<Scalar>(
      { -1.066000,  2.932156,  1.569989 }));
    inverses_.push_back(Pose2<Scalar>(
      { -3.067948, -1.025165,  2.288786 }));
    inverses_.push_back(Pose2<Scalar>(
      { -0.197110,  2.640512,  1.766519 }));
    // initialize inverse jacobians
    JminusType Jminus;
    Jminus <<
        -0.000807,  1.000000,  2.932156,
        -1.000000, -0.000807,  1.066000,
         0.000000,  0.000000, -1.000000;
    inverse_jacobians_.push_back(Jminus);
    Jminus <<
         0.657872,  0.753130, -1.025165,
        -0.753130,  0.657872,  3.067948,
         0.000000,  0.000000, -1.000000;
    inverse_jacobians_.push_back(Jminus);
    Jminus <<
         0.194475,  0.980907,  2.640512,
        -0.980907,  0.194475,  0.197110,
         0.000000,  0.000000, -1.000000;
    inverse_jacobians_.push_back(Jminus);
    // initialize head2tails ( head2tail(poses[i], poses[(i+1)%3]) )
    head2tails_.push_back(Pose2<Scalar>(
      { -5.917290,  0.175458,  2.424410 }));
    head2tails_.push_back(Pose2<Scalar>(
      {  0.724064, -1.216071,  2.227880 }));
    head2tails_.push_back(Pose2<Scalar>(
      { -3.106335,  3.403267,  2.946677 }));
    // initialize head2tail jacobians
    JplusType Jplus;
    Jplus <<
        1.000000,  0.000000, -1.243825,  0.000807,  1.000000,  0.000000,
        0.000000,  1.000000, -2.985996, -1.000000,  0.000807,  0.000000,
        0.000000,  0.000000,  1.000000,  0.000000,  0.000000,  1.000000;
    head2tail_jacobians_.push_back(Jplus);
    Jplus <<
        1.000000,  0.000000, -1.768920, -0.657872,  0.753130,  0.000000,
        0.000000,  1.000000,  1.970298, -0.753130, -0.657872,  0.000000,
        0.000000,  0.000000,  1.000000,  0.000000,  0.000000,  1.000000;
    head2tail_jacobians_.push_back(Jplus);
    Jplus <<
        1.000000,  0.000000, -3.083100, -0.194475,  0.980907,  0.000000,
        0.000000,  1.000000, -0.477904, -0.980907, -0.194475,  0.000000,
        0.000000,  0.000000,  1.000000,  0.000000,  0.000000,  1.000000;
    head2tail_jacobians_.push_back(Jplus);
    // initialize tail2tails ( tail2tail(poses[i], poses[(i+1)%3]) )
    tail2tails_.push_back(Pose2<Scalar>(
      {  1.917984,  1.683513, -0.718797 }));
    tail2tails_.push_back(Pose2<Scalar>(
      { -1.579905, -3.215344,  0.522267 }));
    tail2tails_.push_back(Pose2<Scalar>(
      {  1.420923, -0.027046,  0.196530 }));
    // initialize tail2tail jacobians
    Jplus <<
       -0.000807,  1.000000,  1.683513,  0.000807, -1.000000,  0.000000,
       -1.000000, -0.000807, -1.917984,  1.000000,  0.000807,  0.000000,
        0.000000,  0.000000, -1.000000,  0.000000,  0.000000,  1.000000;
    tail2tail_jacobians_.push_back(Jplus);
    Jplus <<
        0.657872,  0.753130, -3.215344, -0.657872, -0.753130,  0.000000,
       -0.753130,  0.657872,  1.579905,  0.753130, -0.657872,  0.000000,
        0.000000,  0.000000, -1.000000,  0.000000,  0.000000,  1.000000;
    tail2tail_jacobians_.push_back(Jplus);
    Jplus <<
        0.194475,  0.980907, -0.027046, -0.194475, -0.980907,  0.000000,
       -0.980907,  0.194475, -1.420923,  0.980907, -0.194475,  0.000000,
        0.000000,  0.000000, -1.000000,  0.000000,  0.000000,  1.000000;
    tail2tail_jacobians_.push_back(Jplus);
  }
  virtual ~Pose2Test() = default;
  virtual void SetUp() { }
  virtual void TearDown() { }
};

typedef ::testing::Types<float, double> ScalarTypes;
TYPED_TEST_CASE(Pose2Test, ScalarTypes);

TYPED_TEST(Pose2Test, HasExpectedRotationTranslation) {
  for (int i = 0; i < this->kN_; i++) {
    EXPECT_MATRICES_APPROX_EQ(this->poses_[i].so2().matrix(),
                              this->rotation_matrices_[i]);
    EXPECT_MATRICES_APPROX_EQ(this->poses_[i].translation(),
                              this->translations_[i]);
  }
}

TYPED_TEST(Pose2Test, CanConvertToAndFromSE2) {
  for (int i = 0; i < this->kN_; i++) {
    // convert to SE2
    ::Sophus::SE2<TypeParam> pose_as_se2 =
        static_cast<::Sophus::SE2<TypeParam>>(this->poses_[i]);
    EXPECT_MATRICES_APPROX_EQ(this->poses_[i].matrix(),
                              pose_as_se2.matrix());
    // convert back to Pose2
    Pose2<TypeParam> pose_from_se2 =
        static_cast<Pose2<TypeParam>>(pose_as_se2);
    EXPECT_MATRICES_APPROX_EQ(this->poses_[i].matrix(),
                              pose_from_se2.matrix());
  }
}

TYPED_TEST(Pose2Test, PoseRepresentation) {
  for (int i = 0; i < this->kN_; i++) {
    EXPECT_MATRICES_APPROX_EQ(this->poses_[i].xyt(),
                              this->representations_[i]);
  }
}

TYPED_TEST(Pose2Test, HeadToTail) {
  ::Eigen::Matrix<TypeParam, 3, 6> J;
  for (int i = 0; i < this->kN_; i++) {
    Pose2<TypeParam> head2tail =
        this->poses_[i].HeadToTail(this->poses_[(i + 1) % this->kN_], &J);
    EXPECT_MATRICES_APPROX_EQ(head2tail.xyt(),
                              this->head2tails_[i].xyt());
    EXPECT_MATRICES_APPROX_EQ(J, this->head2tail_jacobians_[i]);
  }
}

TYPED_TEST(Pose2Test, TailToTail) {
  ::Eigen::Matrix<TypeParam, 3, 6> J;
  for (int i = 0; i < this->kN_; i++) {
    Pose2<TypeParam> tail2tail =
        this->poses_[i].TailToTail(this->poses_[(i + 1) % this->kN_], &J);
    EXPECT_MATRICES_APPROX_EQ(tail2tail.xyt(),
                              this->tail2tails_[i].xyt());
    EXPECT_MATRICES_APPROX_EQ(J, this->tail2tail_jacobians_[i]);
  }
}

TYPED_TEST(Pose2Test, Inverse) {
  ::Eigen::Matrix<TypeParam, 3, 3> J;
  for (int i = 0; i < this->kN_; i++) {
    Pose2<TypeParam> inverse = this->poses_[i].Inverse(&J);
    EXPECT_MATRICES_APPROX_EQ(inverse.xyt(),
                              this->inverses_[i].xyt());
    EXPECT_MATRICES_APPROX_EQ(J, this->inverse_jacobians_[i]);
  }
}

} // namespace
} // namespace perls_math_cc
