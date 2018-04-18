#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <perls-math/fasttrig.h>
#include <perls-math/so3.h>
#include <perls-math/ssc.h>

// matrix is (r x c)
void print_matrix(const double *M, const int kR, const int kC) {
  int i, j;
  assert(kR >= 1);
  assert(kC >= 1);
  for (i = 0; i < kR; i++) {
    printf("[ ");
    for (j = 0; j < kC-1; j++) {
      printf("%9.6f, ", M[kC*i+j]);
    }
    printf("%9.6f ]\n", M[kC*(i+1)-1]);
  }
}

int main() {

  int i, j;
  double R[9], rph[3], pose[6], Jminus[6*6], Jplus[6*12];

  const int num_poses = 3;
  double **poses;

  // init
  srand(42);
  fasttrig_init();

  // initialize random poses
  poses = calloc(num_poses, sizeof(double *));
  for (i = 0; i < num_poses; i++) {
    poses[i] = calloc(6, sizeof(double));
    for (j = 0; j <= 6; j++) {
      poses[i][j] = (2 * M_PI * rand())/RAND_MAX - M_PI;
    }
    // roll and heading are module 2*pi
    // but pitch must be modulo pi
    poses[i][4] = (M_PI * rand())/RAND_MAX - M_PI_2;
    // uncomment below to test SSC on the plane
    // poses[i][2] = poses[i][3] = poses[i][4] = 0;
  }

  // for each pose
  // operations with multiple poses will cycle
  for (i = 0; i < num_poses; i++) {
    printf("***** poses[%d]: ", i);
    print_matrix(poses[i], 1, 6);

    printf("\n");

    // so3_rotxyz
    so3_rotxyz(R, poses[i] + 3);
    printf("so3_rotxyz(R, rph):\n");
    printf("rph: ");
    print_matrix(poses[i] + 3, 1, 3);
    printf("R:\n");
    print_matrix(R, 3, 3);

    printf("\n");

    // so3_rot2rph
    so3_rot2rph(R, rph);
    printf("so3_rot2rph(R, rph):\n");
    printf("rph: ");
    print_matrix(rph, 1, 3);

    printf("\n");

    // ssc_inverse
    ssc_inverse(pose, Jminus, poses[i]);
    printf("ssc_inverse(X_ji, Jminus, X_ij):\n");
    printf("X_ij: ");
    print_matrix(poses[i], 1, 6);
    printf("X_ji: ");
    print_matrix(pose, 1, 6);
    printf("Jminus:\n");
    print_matrix(Jminus, 6, 6);

    printf("\n");

    // ssc_head2tail
    ssc_head2tail(pose, Jplus, poses[i], poses[(i+1) % num_poses]);
    printf("ssc_head2tail(X_ik, Jplus, X_ij, X_jk):\n");
    printf("X_ij: ");
    print_matrix(poses[i], 1, 6);
    printf("X_jk: ");
    print_matrix(poses[(i+1) % num_poses], 1, 6);
    printf("X_ik: ");
    print_matrix(pose, 1, 6);
    printf("Jplus:\n");
    print_matrix(Jplus, 6, 12);

    printf("\n");

    // ssc_tail2tail
    ssc_tail2tail(pose, Jplus, poses[i], poses[(i+1) % num_poses]);
    printf("ssc_tail2tail(X_jk, Jplus, X_ij, X_ik):\n");
    printf("X_ij: ");
    print_matrix(poses[i], 1, 6);
    printf("X_ik: ");
    print_matrix(poses[(i+1) % num_poses], 1, 6);
    printf("X_jk: ");
    print_matrix(pose, 1, 6);
    printf("Jplus:\n");
    print_matrix(Jplus, 6, 12);

    printf("\n");
  }

  return 0;
}
