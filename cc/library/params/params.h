#pragma once

#include <stddef.h>

namespace library {
namespace params {

// For occupancy grids
extern int kMaxVelodyneScanPoints;
extern float kResolution;
extern float kMaxRange;

extern float kOccGridRangeXY;
extern float kOccGridRangeZ;
extern float kOccGridZ0;
extern int kOccGridMinXY;
extern int kOccGridMaxXY;
extern int kOccGridMinZ;
extern int kOccGridMaxZ;
extern int kOccGridSizeXY;
extern int kOccGridSizeZ;

// For flow computation
//extern float kMaxFlow; // meters per second
extern int kSearchSize;
extern int kMinSearchDist;
extern int kMaxSearchDist;

extern int kPatchSize;

} // params
} // library
