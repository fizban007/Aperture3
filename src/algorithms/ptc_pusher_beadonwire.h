#ifndef _PTC_PUSHER_BEADONWIRE_H_
#define _PTC_PUSHER_BEADONWIRE_H_

#include "data/grid.h"
#include "data/typedefs.h"
#include "particle_pusher.h"
#include <array>

namespace Aperture {

class Environment;

class ParticlePusher_BeadOnWire : public ParticlePusher {
 public:
  typedef ParticlePusher_BeadOnWire self_type;

  ParticlePusher_BeadOnWire(const Environment& env);
  virtual ~ParticlePusher_BeadOnWire();

  virtual void push(cu_sim_data& data, double dt);

  void lorentz_push(Particles& particles, const cu_vector_field<Scalar>& E,
                    const cu_vector_field<Scalar>& B, double dt);
  void move_ptc(Particles& particles, const Grid& grid, double dt);
  void move_photons(Photons& photons, const Grid& grid, double dt);
  void handle_boundary(cu_sim_data& data);
  // void set_interp_order(int order);

  void extra_force(Particles& particles, Index_t idx, double x,
                   const Grid& grid, double dt);
  // virtual void print() { std::cout << "This is particle pusher" <<
  // std::endl; }

 private:
  const SimParams& m_params;
  // int m_order = 3;
  // Interpolator m_interp;
  // bool m_radiation;

};  // ----- end of class ParticlePusher_Geodesic : public
    // ParticlePusher -----

}  // namespace Aperture

#endif  // _PTC_PUSHER_BEADONWIRE_H_
