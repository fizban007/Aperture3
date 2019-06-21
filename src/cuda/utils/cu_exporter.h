#ifndef _CU_EXPORTER_H_
#define _CU_EXPORTER_H_

#include "cuda/data/cu_multi_array.h"
#include "utils/exporter.h"

namespace Aperture {

struct cu_sim_data;
class cu_sim_environment;

struct cu_dataset {
  std::string name;
  cu_multi_array<float> f;
};

class cu_exporter : public exporter {
 public:
  cu_exporter(cu_sim_environment& env, uint32_t& timestep);
  virtual ~cu_exporter();

  void write_output(cu_sim_data& data);

 private:
  template <typename Func>
  void add_grid_output(cu_sim_data& data, const std::string& name, Func f);

  template <typename Func>
  void add_array_output(cu_sim_data& data, const std::string& name, Func f);

  cu_dataset m_tmp_cudata;
  cu_multi_array<float> tmp_grid_cudata;
};

}  // namespace Aperture

#endif  // _CU_EXPORTER_H_
