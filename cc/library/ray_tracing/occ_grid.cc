#include "library/ray_tracing/occ_grid.h"

#include <algorithm>
#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>

namespace library {
namespace ray_tracing {

OccGrid::OccGrid(const OccGridData &ogd) :
 OccGrid(ogd.locations, ogd.log_odds, ogd.resolution) {
}

float OccGrid::GetLogOdds(const Location &loc) const {
  std::vector<Location>::const_iterator it = std::lower_bound(data_.locations.begin(), data_.locations.end(), loc);
  if (it != data_.locations.end() && (*it) == loc) {
    size_t pos = it - data_.locations.begin();
    return data_.log_odds[pos];
  }

  // Unknown
  return 0.0f;
}

float OccGrid::GetProbability(const Location &loc) const {
  double lo = GetLogOdds(loc);
  return 1 / (1 + exp(-lo));
}

float OccGrid::GetLogOdds(float x, float y, float z) const {
  Location loc(x, y, z, data_.resolution);
  return GetLogOdds(loc);
}

float OccGrid::GetProbability(float x, float y, float z) const {
  double lo = GetLogOdds(x, y, z);
  return 1 / (1 + exp(-lo));
}

const std::vector<Location>& OccGrid::GetLocations() const {
  return data_.locations;
}

const std::vector<float>& OccGrid::GetLogOdds() const {
  return data_.log_odds;
}

float OccGrid::GetResolution() const {
  return data_.resolution;
}

std::map<Location, float> OccGrid::MakeMap() const {
  std::map<Location, float> og_map;

  for (size_t i = 0; i < data_.locations.size(); i++) {
    float lo = data_.log_odds[i];
    og_map[data_.locations[i]] = lo > 0 ? 1:0;
  }

  return og_map;
}

void OccGrid::Save(const char* fn) const {
  std::ofstream ofs(fn);
  boost::archive::binary_oarchive oa(ofs);
  oa << data_;
}

OccGrid OccGrid::Load(const char* fn) {
  OccGridData data;
  std::ifstream ifs(fn);
  boost::archive::binary_iarchive ia(ifs);
  ia >> data;

  return OccGrid(data);
}

}  // namespace ray_tracing
}  // namespace library
