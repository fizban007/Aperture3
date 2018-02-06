#ifndef _PTC_PUSHER_GEODESIC_H_
#define _PTC_PUSHER_GEODESIC_H_

#include "algorithms/interpolation.h"
#include "data/typedefs.h"
#include "particle_pusher.h"
#include <array>

namespace Aperture {

class ParticlePusher_Geodesic : public ParticlePusher {
 public:
  typedef ParticlePusher_Geodesic self_type;

  ParticlePusher_Geodesic();
  virtual ~ParticlePusher_Geodesic();

  virtual void push(SimData& data, double dt);

  void lorentz_push(Particles& particles, Index_t idx, double x,
                    const VectorField<Scalar>& E, const VectorField<Scalar>& B,
                    double dt);
  void move_ptc(Particles& particles, Index_t idx, double x, const Grid& grid,
                double dt);

  void set_interp_order(int order);

  // virtual void print() { std::cout << "This is particle pusher" << std::endl; }

 private:
  int m_order = 3;
  // Interpolator m_interp;
  bool m_radiation;

};  // ----- end of class ParticlePusher_Geodesic : public ParticlePusher -----

}  // namespace Aperture

#endif  // _PTC_PUSHER_GEODESIC_H_
