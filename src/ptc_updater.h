#ifndef _PTC_UPDATER_H_
#define _PTC_UPDATER_H_

namespace Aperture {

struct SimData;
class Environment;

class PtcUpdater
{
 public:
  PtcUpdater(const Environment& env);
  virtual ~PtcUpdater();

  void update_particles(SimData& data, double dt);
  void handle_boundary(SimData& data);

 private:
  const Environment& m_env;

  cudaPitchedPtr m_E1, m_E2, m_E3;
  cudaPitchedPtr m_B1, m_B2, m_B3;
  cudaPitchedPtr m_J1, m_J2, m_J3;
  cudaPitchedPtr* m_Rho;
  Extent m_extent;
}; // ----- end of class PtcUpdater -----

}

#endif  // _PTC_UPDATER_H_
