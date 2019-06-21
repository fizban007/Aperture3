#ifndef _EXPORTER_H_
#define _EXPORTER_H_

#include "core/multi_array.h"
#include <fstream>

namespace Aperture {

struct sim_data;
class sim_environment;

struct dataset {
  std::string name;
  multi_array<float> f;
};

class exporter {
 public:
  exporter(sim_environment& env, uint32_t& timestep);
  virtual ~exporter();

  template <typename Func>
  void add_grid_output(sim_data& data, const std::string& name, Func& f);

 protected:
  // std::unique_ptr<Grid> grid;
  sim_environment& m_env;
  std::string
      outputDirectory;  //!< Sets the directory of all the data files
  std::string subDirectory;  //!< Sets the directory of current rank
  std::string subName;
  std::string filePrefix;  //!< Sets the common prefix of the data files

  std::ofstream xmf;

  dataset m_tmp_data;
  dataset m_grid_data;
};

}  // namespace Aperture

#endif  // _EXPORTER_H_
