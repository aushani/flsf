#include "library/kitti/object_label.h"

#include <iostream>

namespace library {
namespace kitti {

ObjectLabels ObjectLabel::Load(const char *fn) {
  ObjectLabels labels;

  FILE *fp = fopen(fn, "r");

  char type[100];
  char *line = NULL;
  size_t len = 0;
  while (getline(&line, &len, fp) != -1) {
    //printf("read line: <%s>\n", line);
    ObjectLabel label;
    sscanf(line, "%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
        type,
        &label.truncated,
        &label.occluded,
        &label.alpha,
        &label.bbox[0],
        &label.bbox[1],
        &label.bbox[2],
        &label.bbox[3],
        &label.dimensions[0],
        &label.dimensions[1],
        &label.dimensions[2],
        &label.location[0],
        &label.location[1],
        &label.location[2],
        &label.rotation_y);

    label.type = GetType(type);
    label.ComputeTransforms();

    labels.push_back(label);
  }

  fclose(fp);

  return labels;
}

void ObjectLabel::Save(const ObjectLabels &labels, const char *fn) {
  FILE *fp = fopen(fn, "w");

  for (const auto &label : labels) {
    fprintf(fp, "%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
        GetString(label.type),
        label.truncated,
        label.occluded,
        label.alpha,
        label.bbox[0],
        label.bbox[1],
        label.bbox[2],
        label.bbox[3],
        label.dimensions[0],
        label.dimensions[1],
        label.dimensions[2],
        label.location[0],
        label.location[1],
        label.location[2],
        label.rotation_y,
        label.score);

  }

  fclose(fp);
}

ObjectLabel::Type ObjectLabel::GetType(const char *type) {
  size_t count = strlen(type);
  if (strncmp(type, "Car", count) == 0) {
    return CAR;
  } else if (strncmp(type, "Van", count) == 0) {
    return VAN;
  } else if (strncmp(type, "Truck", count) == 0) {
    return TRUCK;
  } else if (strncmp(type, "Pedestrian", count) == 0) {
    return PEDESTRIAN;
  } else if (strncmp(type, "Person_sitting", count) == 0) {
    return PERSON_SITTING;
  } else if (strncmp(type, "Cyclist", count) == 0) {
    return CYCLIST;
  } else if (strncmp(type, "Tram", count) == 0) {
    return TRAM;
  } else if (strncmp(type, "Misc", count) == 0) {
    return MISC;
  } else if (strncmp(type, "DontCare", count) == 0) {
    return DONT_CARE;
  } else {
    BOOST_ASSERT(false);
    return DONT_CARE;
  }
}

const char* ObjectLabel::GetString(const Type &type) {
  switch(type) {
    case CAR:
      return "Car";
    case VAN:
      return "Van";
    case TRUCK:
      return "Truck";
    case PEDESTRIAN:
      return "Pedestrian";
    case PERSON_SITTING:
      return "Person_sitting";
    case CYCLIST:
        return "Cyclist";
    case TRAM:
      return "Tram";
    case MISC:
      return "Misc";
    case DONT_CARE:
      return "DontCare";
  }

  return "DontCare";
}

void ObjectLabel::LoadCalib(const char *fn, Eigen::Matrix<double, 3, 4> *p, Eigen::Matrix4d *r, Eigen::Matrix4d *t_cv) {
  FILE *f_cc = fopen(fn, "r");

  const char *header_p = "P2: "; // according to dascar according to kitti paper
  const char *header_r = "R0_rect: ";
  const char *header_t_cv = "Tr_velo_to_cam: ";

  char *line = NULL;
  size_t len = 0;

  double mat_r[9];
  double mat_p[12];
  double mat_tcv[12];

  while (getline(&line, &len, f_cc) != -1) {
    if (strncmp(header_r, line, strlen(header_r)) == 0) {
      sscanf(&line[strlen(header_r)], "%lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
              &mat_r[0], &mat_r[1], &mat_r[2], &mat_r[3], &mat_r[4], &mat_r[5], &mat_r[6], &mat_r[7], &mat_r[8]);
    } else if (strncmp(header_p, line, strlen(header_p)) == 0) {
      sscanf(&line[strlen(header_p)], "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
              &mat_p[0], &mat_p[1], &mat_p[2], &mat_p[3], &mat_p[4], &mat_p[5], &mat_p[6], &mat_p[7], &mat_p[8], &mat_p[9], &mat_p[10], &mat_p[11]);
    } else if (strncmp(header_t_cv, line, strlen(header_t_cv)) == 0) {
      sscanf(&line[strlen(header_t_cv)],
          "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
          &mat_tcv[0], &mat_tcv[1], &mat_tcv[2], &mat_tcv[3], &mat_tcv[4], &mat_tcv[5], &mat_tcv[6],
          &mat_tcv[7], &mat_tcv[8], &mat_tcv[9], &mat_tcv[10], &mat_tcv[11]);
    }
  }

  fclose(f_cc);

  t_cv->setZero();
  for (int i=0; i<3; i++) {
    for (int j=0; j<4; j++) {
      (*t_cv)(i, j) = mat_tcv[i*4 + j];
    }
  }
  (*t_cv)(3, 3) = 1;

  p->setZero();
  for (int i=0; i<3; i++) {
    for (int j=0; j<4; j++) {
      (*p)(i, j) = mat_p[i*4 + j];
    }
  }

  r->setZero();
  for (int i=0; i<3; i++) {
    for (int j=0; j<3; j++) {
      (*r)(i, j) = mat_r[i*3 + j];
    }
  }
  (*r)(3, 3) = 1.0;

}

} // namespace kitti
} // namespace library
