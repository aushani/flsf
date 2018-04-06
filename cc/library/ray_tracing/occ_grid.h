// Adapted from dascar
#pragma once

#include <map>
#include <vector>

#include <boost/assert.hpp>

#include "library/ray_tracing/occ_grid_location.h"

namespace library {
namespace ray_tracing {

// Helper data struct to make serialization easier while still maintaining constness of OccGrid
struct OccGridData {
  // These are parallel containers that need to be sorted and the same size
  std::vector<Location> locations;
  std::vector<float> log_odds;

  float resolution = 0.0;

  OccGridData() {;}

  OccGridData(const std::vector<Location> &locs, const std::vector<float> &los, float res) :
    locations(locs), log_odds(los), resolution(res) {
  }

  template<class Archive>
  void serialize(Archive & ar, const unsigned int /* file_version */){
    ar & locations;
    ar & log_odds;
    ar & resolution;
  }
};

// This struct represents an occupancy grid, holding log odds values for the
// probablity that a given voxel is occupied. Postive means occupied, negative means
// free, and 0 means unknown.
class OccGrid {
 public:
  // locations and log_odds must be sorted and the same size
  OccGrid(const std::vector<Location> &locs, const std::vector<float> &los, float res) : data_(locs, los, res) {
    BOOST_ASSERT(locs.size() == los.size());
    BOOST_ASSERT(res > 0);
  }

  OccGrid(const OccGridData &ogd);

  float GetProbability(const Location &loc) const;
  float GetProbability(float x, float y, float z) const;

  float GetLogOdds(const Location &loc) const;
  float GetLogOdds(float x, float y, float z) const;

  const std::vector<Location>& GetLocations() const;
  const std::vector<float>& GetLogOdds() const;
  float GetResolution() const;

  std::map<Location, float> MakeMap() const;

  void Save(const char* fn) const;
  static OccGrid Load(const char* fn);

 protected:
  OccGrid() {}

  OccGridData data_;
};

}  // namespace ray_tracing
}  // namespace library
