#ifndef PERLSMATHCC_TESTUTIL_H_
#define PERLSMATHCC_TESTUTIL_H_

// gtest macros to test if Eigen matrices are the same
#define EXPECT_MATRICES_APPROX_EQ(M_actual, M_expected) \
  EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-3)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected
#define ASSERT_MATRICES_APPROX_EQ(M_actual, M_expected) \
  ASSERT_TRUE(M_actual.isApprox(M_expected, 1e-3)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected

#endif // PERLSMATHCC_TESTUTIL_H_
