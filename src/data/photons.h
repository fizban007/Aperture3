#ifndef _PHOTONS_H_
#define _PHOTONS_H_

#include "data/particle_base.h"
#include "utils/util_functions.h"

namespace Aperture {

struct SimParams;

class photons_t : public particle_base<single_photon_t> {
 public:
  typedef particle_base<single_photon_t> base_class;
  typedef photon_data data_class;

  photons_t();
  photons_t(size_t max_num);
  photons_t(const SimParams& params);
  photons_t(const photons_t& other);
  photons_t(photons_t&& other);
  virtual ~photons_t();

  using base_class::append;
  using base_class::put;
  void append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p, int cell,
              Scalar path_left, Scalar weight = 1.0, uint32_t flag = 0);

  void track(Index_t pos) {
    m_data.flag[pos] |= bit_or(PhotonFlag::tracked);
  }
  bool check_flag(Index_t pos, PhotonFlag flag) const {
    return (m_data.flag[pos] & bit_or(flag)) == bit_or(flag);
  }
  void set_flag(Index_t pos, PhotonFlag flag) {
    m_data.flag[pos] |= bit_or(flag);
  }
};  // ----- end of class photons_t : public
    // particle_base<single_photon_t>  -----

}  // namespace Aperture

#endif  // _PHOTONS_H_
