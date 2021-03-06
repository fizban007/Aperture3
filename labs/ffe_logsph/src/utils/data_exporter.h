#ifndef _DATA_EXPORTER_H_
#define _DATA_EXPORTER_H_

#include "core/multi_array.h"
#include "hdf5.h"
#include <boost/multi_array.hpp>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

namespace Aperture {

struct sim_data;
class sim_environment;

class data_exporter {
 public:
  data_exporter(sim_environment& env, uint32_t& timestep);
  virtual ~data_exporter();

  void write_grid();
  void copy_config_file();
  void write_xmf_head(std::ofstream& fs);
  void write_xmf_step_header(std::ofstream& fs, double time);
  void write_xmf_step_header(std::string& buffer, double time);
  void write_xmf_step_close(std::ofstream& fs);
  void write_xmf_step_close(std::string& buffer);
  void write_xmf_tail(std::ofstream& fs);
  void write_xmf_tail(std::string& buffer);
  void write_xmf(uint32_t step, double time);
  void prepare_xmf_restart(uint32_t restart_step, int data_interval,
                           float time);
  void write_snapshot(sim_data& data, uint32_t step);
  void load_from_snapshot(sim_data& data, uint32_t step, double time);

  void write_output(sim_data& data, uint32_t timestep, double time);

  void write_field_output(sim_data& data, uint32_t timestep,
                          double time);
  void write_ptc_output(sim_data& data, uint32_t timestep, double time);

  void write_multi_array(const multi_array<float>& array,
                         const std::string& name,
                         const Extent& total_ext, const Index& offset,
                         hid_t file_id);

  void write_snapshot(const sim_data& data, const std::string& filename);

  template <typename T>
  void write_collective_array(const T* array, const std::string& name,
                              size_t total, size_t local, size_t offset,
                              hid_t file_id);

  void add_array_output(multi_array<float>& array,
                        const std::string& name, hid_t file_id,
                        uint32_t timestep);

  template <typename Func>
  void add_grid_output(sim_data& data, const std::string& name, Func f,
                       hid_t file_id, uint32_t timestep);

  void add_ptc_output(sim_data& data, int species, hid_t file_id,
                      uint32_t timestep);

  template <typename Func>
  void add_ptc_float_output(sim_data& data, const std::string& name,
                            uint64_t num, uint64_t total,
                            uint64_t offset, Func f, hid_t file_id,
                            uint32_t timestep);

  template <typename Func>
  void add_ptc_uint_output(sim_data& data, const std::string& name,
                           uint64_t num, uint64_t total,
                           uint64_t offset, Func f, hid_t file_id,
                           uint32_t timestep);

 protected:
  // std::unique_ptr<Grid> grid;
  sim_environment& m_env;
  std::string
      outputDirectory;  //!< Sets the directory of all the data files

  std::ofstream m_xmf;  //!< This is the accompanying xmf file
                        //!< describing the hdf structure
  std::string m_dim_str;
  std::string m_xmf_buffer;

  multi_array<float> tmp_grid_data;  //!< This stores the temporary
                                     //!< downsampled data for output
  // boost::multi_array<float, 3> m_output_3d;
  // boost::multi_array<float, 2> m_output_2d;
  // std::vector<float> m_output_1d;

  // std::vector<float> tmp_ptc_float_data;
  // std::vector<uint32_t> tmp_ptc_uint_data;

  // std::unique_ptr<std::thread> m_fld_thread;
  // std::unique_ptr<std::thread> m_ptc_thread;
};

}  // namespace Aperture

#endif  // _DATA_EXPORTER_H_
