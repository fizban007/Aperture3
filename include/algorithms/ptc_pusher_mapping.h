#ifndef _PTC_PUSHER_MAPPING_H_
#define _PTC_PUSHER_MAPPING_H_

#include "particle_pusher.h"
#include "algorithms/forces.h"
#include "algorithms/interpolation.h"
#include "metrics.h"

namespace Aperture {

class ParticlePusher_Mapping : public ParticlePusher
{
 public:
  typedef ParticlePusher_Mapping self_type;

  ParticlePusher_Mapping(int interp_order = 1);
  virtual ~ParticlePusher_Mapping();

  virtual void push(SimData& data, double dt);
  virtual void push(Particles& particles, const vfield& E, const vfield& B, double dt);

  void interpolate_field (Particles& particles, const vfield& E, const vfield& B);
  void update_momentum (Particles& particles, const vfield& E, const vfield& B, double dt);
  void update_momentum (Particles& particles, const vfield& E, const vfield& B, double dt, Index_t start, Index_t num);
  void update_momentum_avx (Particles& particles, const vfield& E, const vfield& B, double dt);

  struct update_position_f {
    template <typename Metric>
    void operator() (const Metric& metric, Particles& particles, const Grid& grid, double dt, Index_t start = 0, Index_t num = 0) const;
  } update_position;

  struct update_position_avx_f {
    template <typename Metric>
    void operator() (const Metric& metric, ParticlePusher_Mapping& pusher, Particles& particles, const Grid& grid, double dt) const;
  } update_position_avx;
  // void update_position (Particles&)

  Vec3<Mom_t> boris_push(const Vec3<Mom_t>& p, const Vec3<Scalar>& E, const Vec3<Scalar>& B,
                         double q_over_m, double dt);
  Vec3<Mom_t> vay_push(const Vec3<Mom_t>& p, const Vec3<Scalar>& E, const Vec3<Scalar>& B,
                       double q_over_m, double dt);

#if defined(__AVX2__) && (defined(__ICC) || defined(__INTEL_COMPILER))
  void vay_push_avx(__m256d* p1, __m256d* p2, __m256d* p3, __m256d* gamma,
                    __m256d E1, __m256d E2, __m256d E3,
                    __m256d B1, __m256d B2, __m256d B3,
                    double q_over_m, double dt);
#endif

 private:
  int m_order;
  Interpolator m_interp;

  // radiative_force m_rad;
}; // ----- end of class particle_pusher_mapping : public particle_pusher -----


}

#include "algorithms/ptc_pusher_mapping_impl.hpp"

#endif  // _PTC_PUSHER_MAPPING_H_
