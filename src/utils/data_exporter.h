#ifndef _DATA_EXPORTER_H_
#define _DATA_EXPORTER_H_

#include "utils/hdf_exporter.h"

namespace Aperture {

class sim_environment;
struct sim_data;
template <typename T>
class scalar_field;
template <typename T>
class vector_field;

class data_exporter : public hdf_exporter {
 public:
  data_exporter(SimParams& params, uint32_t& timestep);
  virtual ~data_exporter();

  template <typename T>
  void add_field(const std::string& name, scalar_field<T>& field);
  template <typename T>
  void add_field(const std::string& name, vector_field<T>& field);

  void write_snapshot(sim_environment& env, sim_data& data,
                     uint32_t timestep);
  void load_from_snapshot(sim_environment& env, sim_data& data,
                          uint32_t& timestep);

};  // ----- end of class cu_data_exporter : public hdf_exporter -----

}  // namespace Aperture

#endif  // _DATA_EXPORTER_H_
