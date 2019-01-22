#ifndef _CU_DATA_EXPORTER_H_
#define _CU_DATA_EXPORTER_H_

#include "utils/hdf_exporter.h"

namespace Aperture {

class cu_sim_environment;
struct cu_sim_data;
template <typename T>
class cu_scalar_field;
template <typename T>
class cu_vector_field;

class cu_data_exporter : public hdf_exporter<cu_data_exporter> {
 public:
  cu_data_exporter(SimParams& params, uint32_t& timestep);
  virtual ~cu_data_exporter();

  template <typename T>
  void add_field(const std::string& name, cu_scalar_field<T>& field,
                 bool sync = true);
  template <typename T>
  void add_field(const std::string& name, cu_vector_field<T>& field,
                 bool sync = true);

  void write_snapshot(cu_sim_environment& env, cu_sim_data& data,
                     uint32_t timestep);
  void load_from_snapshot(cu_sim_environment& env, cu_sim_data& data,
                          uint32_t& timestep);

  template <typename T>
  void interpolate_field_values(fieldoutput<1>& field, int components, const T& t);
  template <typename T>
  void interpolate_field_values(fieldoutput<2>& field, int components, const T& t);
  template <typename T>
  void interpolate_field_values(fieldoutput<3>& field, int components, const T& t);

};  // ----- end of class cu_data_exporter : public hdf_exporter -----


} // namespace Aperture


#endif  // _CU_DATA_EXPORTER_H_
