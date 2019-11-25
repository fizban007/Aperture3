#ifndef _PHOTONS_H_
#define _PHOTONS_H_

#include "core/particle_base.h"
#include "utils/util_functions.h"

namespace Aperture {

struct SimParams;

class photons_t : public particle_base<single_photon_t> {
 public:
  typedef particle_base<single_photon_t> base_class;
  typedef photon_data data_class;

  photons_t();
  photons_t(size_t max_num);
  photons_t(const photons_t& other);
  photons_t(photons_t&& other);
  virtual ~photons_t();

  using base_class::append;
  using base_class::put;
  void append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p, int cell,
              Scalar path_left, Scalar weight = 1.0, uint32_t flag = 0);
};  // ----- end of class photons_t : public
    // particle_base<single_photon_t>  -----

}  // namespace Aperture

#endif  // _PHOTONS_H_
