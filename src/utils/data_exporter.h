#ifndef _DATA_EXPORTER_H_
#define _DATA_EXPORTER_H_

#include "core/multi_array.h"
#include <boost/multi_array.hpp>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

namespace HighFive {

class File;

}

namespace Aperture {

struct sim_data;
class sim_environment;

class data_exporter {
 public:
  data_exporter(sim_environment& env, uint32_t& timestep);
  virtual ~data_exporter();

  void write_output(sim_data& data, uint32_t timestep, double time);

  void write_field_output(sim_data& data, uint32_t timestep,
                          double time);
  void write_ptc_output(sim_data& data, uint32_t timestep, double time);

 protected:
  void add_array_output(multi_array<float>& array,
                        const std::string& name, HighFive::File& file);

  template <typename Func>
  void add_grid_output(sim_data& data, const std::string& name, Func f,
                       HighFive::File& file);

  template <typename Func>
  void add_ptc_float_output(sim_data& data, const std::string& name,
                            Func f, HighFive::File& file);

  template <typename Func>
  void add_ptc_uint_output(sim_data& data, const std::string& name,
                           Func f, HighFive::File& file);

  // std::unique_ptr<Grid> grid;
  sim_environment& m_env;
  std::string
      outputDirectory;  //!< Sets the directory of all the data files

  std::ofstream xmf;  //!< This is the accompanying xmf file describing
                      //!< the hdf structure

  multi_array<float> tmp_grid_data;  //!< This stores the temporary
                                     //!< downsampled data for output
  boost::multi_array<float, 3> m_output_3d;
  boost::multi_array<float, 2> m_output_2d;
  std::vector<float> m_output_1d;

  std::vector<float> tmp_ptc_float_data;
  std::vector<uint32_t> tmp_ptc_uint_data;

  // std::unique_ptr<std::thread> m_fld_thread;
  // std::unique_ptr<std::thread> m_ptc_thread;
};

}  // namespace Aperture

#endif  // _DATA_EXPORTER_H_
