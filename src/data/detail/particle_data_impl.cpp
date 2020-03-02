#ifndef _PARTICLE_DATA_IMPL_H_
#define _PARTICLE_DATA_IMPL_H_

#include "data/particle_data.h"
#include "utils/for_each_arg.hpp"
#include <cstdlib>

namespace Aperture {

struct read_at_idx {
  size_t idx_;
  HOST_DEVICE read_at_idx(size_t idx) : idx_(idx) {}

  template <typename T, typename U>
  HD_INLINE void operator()(T& t, U& u) const {
    u = t[idx_];
  }
};

single_particle_t particle_data::operator[](size_t idx) const {
  single_particle_t part;
  for_each_arg(*this, part, read_at_idx(idx));
  return part;
}

single_photon_t photon_data::operator[](size_t idx) const {
  single_photon_t part;
  for_each_arg(*this, part, read_at_idx(idx));
  return part;
}

single_particle1d_t particle1d_data::operator[](size_t idx) const {
  single_particle1d_t part;
  for_each_arg(*this, part, read_at_idx(idx));
  return part;
}

single_photon1d_t photon1d_data::operator[](size_t idx) const {
  single_photon1d_t part;
  for_each_arg(*this, part, read_at_idx(idx));
  return part;
}
}  // namespace Aperture

#endif  // _PARTICLE_DATA_IMPL_H_
