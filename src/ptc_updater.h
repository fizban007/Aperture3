#ifndef _PTC_UPDATER_H_
#define _PTC_UPDATER_H_

namespace Aperture {

struct SimData;
class Environment;

struct fields_data {
  cudaPitchedPtr E1, E2, E3;
  cudaPitchedPtr B1, B2, B3;
  cudaPitchedPtr J1, J2, J3;
  cudaPitchedPtr* Rho;
};

class PtcUpdater {
 public:
  PtcUpdater(const Environment& env);
  virtual ~PtcUpdater();

  virtual void update_particles(SimData& data, double dt);
  virtual void handle_boundary(SimData& data);

 protected:
  void initialize_dev_fields(SimData& data);
  
  const Environment& m_env;

  fields_data m_dev_fields;
  Extent m_extent;
  bool m_fields_initialized;
};  // ----- end of class PtcUpdater -----

}  // namespace Aperture

#endif  // _PTC_UPDATER_H_
