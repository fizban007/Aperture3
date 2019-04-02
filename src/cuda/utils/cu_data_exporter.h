#ifndef _CU_DATA_EXPORTER_H_
#define _CU_DATA_EXPORTER_H_

#include "utils/hdf_exporter.h"
#include "cuda/data/particles_dev.h"
#include "cuda/data/photons_dev.h"
#include "cuda/data/cu_multi_array.h"

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
  void add_field(const std::string& name,
                 std::vector<cu_scalar_field<T>>& field,
                 bool sync = true);
  template <typename T>
  void add_field(const std::string& name,
                 std::vector<cu_vector_field<T>>& field,
                 bool sync = true);

  void add_ptc_output_1d(const std::string& name,
                         const std::string& type,
                         std::vector<Particles>& ptc);
  void add_ptc_output(const std::string& name, const std::string& type,
                      std::vector<Particles>& ptc);
  void add_ptc_output_1d(const std::string& name,
                         const std::string& type,
                         std::vector<Photons>& ptc);
  void add_ptc_output(const std::string& name, const std::string& type,
                      std::vector<Photons>& ptc);

  void write_snapshot(cu_sim_environment& env, cu_sim_data& data,
                      uint32_t timestep);
  void load_from_snapshot(cu_sim_environment& env, cu_sim_data& data,
                          uint32_t& timestep);
  void write_particles(cu_sim_data& data, uint32_t step, double time);
  void write_output(cu_sim_data& data, uint32_t timestep, double time);
  void set_mesh(cu_sim_data& data);
  void prepare_output(cu_sim_data& data);
  void writeXMFStep(std::ofstream &f, uint32_t step, double time);

  template <int n>
  struct cu_fieldoutput {
    std::string name;
    std::string type;
    std::vector<field_base*> field;
    std::vector<boost::multi_array<float, n>> f;
    bool sync;
  };

  template <typename T, int n>
  struct cu_arrayoutput {
    std::string name;
    std::vector<cu_multi_array<T>*> array;
    boost::multi_array<float, n> f;
  };

  struct cu_ptcoutput_1d {
    std::string name;
    std::string type;
    std::vector<particle_interface*> ptc;
    std::vector<float> x;
    std::vector<float> p;
  };

  struct cu_ptcoutput {
    std::string name;
    std::string type;
    std::vector<particle_interface*> ptc;
    std::vector<float> x1;
    std::vector<float> x2;
    std::vector<float> x3;
    std::vector<float> p1;
    std::vector<float> p2;
    std::vector<float> p3;
  };

  template <typename T>
  void interpolate_field_values(fieldoutput<1>& field, int components,
                                const T& t);
  template <typename T>
  void interpolate_field_values(fieldoutput<2>& field, int components,
                                const T& t);
  template <typename T>
  void interpolate_field_values(fieldoutput<3>& field, int components,
                                const T& t);

 private:
  void add_cu_field_output(const std::string& name,
                           const std::string& type, int num_components,
                           std::vector<field_base*>& field, int dim,
                           bool sync);

  std::vector<cu_fieldoutput<1>> m_fields_1d;
  std::vector<cu_fieldoutput<2>> m_fields_2d;
  std::vector<cu_fieldoutput<3>> m_fields_3d;
  std::vector<cu_arrayoutput<float, 2>> m_float_2d;
  std::vector<cu_ptcoutput> m_ptcdata;
  std::vector<cu_ptcoutput_1d> m_ptcdata_1d;
  std::vector<Quadmesh> m_submesh;
  std::vector<Quadmesh> m_submesh_out;
};  // ----- end of class cu_data_exporter : public hdf_exporter -----

}  // namespace Aperture

#endif  // _CU_DATA_EXPORTER_H_
