#ifndef _PTC_PUSHER_GEODESIC_H_
#define _PTC_PUSHER_GEODESIC_H_

#include "algorithms/interpolation.h"
#include "data/typedefs.h"
#include "particle_pusher.h"
#include <array>

namespace Aperture {

struct ParticleCache {
  bool valid = false;
  double alpha, det;
  double g11, g22, g33, g13;
  double invg11, invg22, invg33, invg13;
  double d1a, d2a, d1b1, d2b1;
  double d1g11, d1g22, d1g33, d1g13;
  double d2g11, d2g22, d2g33, d2g13;
};

class ParticlePusher_Geodesic : public ParticlePusher {
 public:
  typedef ParticlePusher_Geodesic self_type;

  ParticlePusher_Geodesic();
  virtual ~ParticlePusher_Geodesic();

  virtual void push(SimData& data, double dt);

  void lorentz_push(Particles& particles, Index_t idx, double x,
                    const VectorField<Scalar>& E, const VectorField<Scalar>& B,
                    double dt);
  void move(Particles& particles, Index_t idx, double x, const Grid& grid,
                     double dt);

  void set_interp_order(int order);

 private:
  int m_order = 1;
  // Interpolator m_interp;
  bool m_radiation;
  ParticleCache m_cache;
  std::array<std::array<double, 3>, 3> m_boris_mat;

};  // ----- end of class ParticlePusher_Geodesic : public ParticlePusher -----

}  // namespace Aperture

#endif  // _PTC_PUSHER_GEODESIC_H_
