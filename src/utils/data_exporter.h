#ifndef _DATA_EXPORTER_H_
#define _DATA_EXPORTER_H_

#include "core/multi_array.h"
#include "utils/hdf_wrapper.h"
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
  void write_output(sim_data& data, uint32_t timestep, double time);

  void write_field_output(sim_data& data, uint32_t timestep,
                          double time);
  void write_ptc_output(sim_data& data, uint32_t timestep, double time);

  void write_multi_array(const multi_array<float>& array,
                         const std::string& name,
                         const Extent& total_ext, const Index& offset,
                         H5File& file);

  template <typename Func>
  void add_grid_output(sim_data& data, const std::string& name, Func f,
                       H5File& file, uint32_t timestep);

  template <typename T>
  void add_grid_output(multi_array<T>& array, Stagger stagger,
                       const std::string& name, H5File& file,
                       uint32_t timestep);

  template <typename Ptc>
  void add_ptc_output(Ptc& data, size_t num, H5File& file,
                      const std::string& prefix);

  template <typename Ptc>
  void read_ptc_output(Ptc& data, size_t num, H5File& file,
                       const std::string& prefix);

  template <typename T, typename Func>
  void add_tracked_ptc_output(sim_data& data, int sp,
                              const std::string& name,
                              uint64_t total, uint64_t offset,
                              Func f, H5File& file);

  void save_snapshot(const std::string& filename, sim_data& data,
                     uint32_t step, Scalar time);
  void load_snapshot(const std::string& filename, sim_data& data,
                     uint32_t& step, Scalar& time);

 protected:
  sim_environment& m_env;
  std::string
      outputDirectory;  //!< Sets the directory of all the data files

  std::ofstream m_xmf;  //!< This is the accompanying xmf file
                        //!< describing the hdf structure
  std::string m_dim_str;
  std::string m_xmf_buffer;

  multi_array<float> tmp_grid_data;  //!< This stores the temporary
                                     //!< downsampled data for output
  Extent m_out_ext;
  void* tmp_ptc_data;
};

}  // namespace Aperture

#endif  // _DATA_EXPORTER_H_
