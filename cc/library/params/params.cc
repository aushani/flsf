#include "library/params/params.h"

namespace library {
namespace params {

// For occupancy grids
int kMaxVelodyneScanPoints = 150000;
float kResolution = 0.3;
float kMaxRange = 100.0;

float kOccGridRangeXY = 50.0;
float kOccGridRangeZ = 4.2;
float kOccGridZ0 = -0.6;
int kOccGridMinXY = -kOccGridRangeXY / (2 * kResolution);
int kOccGridMaxXY =  kOccGridRangeXY / (2 * kResolution);
int kOccGridMinZ  = (kOccGridZ0 - kOccGridRangeZ/2.0) / kResolution;
int kOccGridMaxZ  = (kOccGridZ0 + kOccGridRangeZ/2.0) / kResolution;

int kOccGridSizeXY = kOccGridMaxXY - kOccGridMinXY + 1;
int kOccGridSizeZ = kOccGridMaxZ - kOccGridMinZ + 1;

// For flow computation
//float kMaxFlow = 45.0; // meters per second
int kSearchSize = 31;
int kMinSearchDist = -kSearchSize / 2;
int kMaxSearchDist =  kSearchSize / 2;

int kPatchSize = 13;

int kPadding = 24;

} // params
} // library
