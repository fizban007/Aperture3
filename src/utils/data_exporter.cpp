#include "H5Cpp.h"
#include "core/constant_defs.h"
#include "data_exporter.h"
#include "sim_params.h"
#include "sim_data.h"
#include "sim_environment.h"

#define ADD_GRID_OUTPUT(input, name, func, file)               \
  add_grid_output(input, name,                                 \
                  [](sim_data & data, multi_array<Scalar> & p, \
                     Index idx, Index idx_out) func,           \
                  file)

using namespace H5;

namespace Aperture {

template <typename Func>
void
sample_grid_quantity2d(sim_data& data, const Grid& g, int downsample,
                       multi_array<Scalar>& result, Func f) {
  const auto& ext = g.extent();
  auto& mesh = g.mesh();
  for (int j = 0; j < ext.height(); j++) {
    for (int i = 0; i < ext.width(); i++) {
      Index idx_out(i, j, 0);
      Index idx_data(i * downsample + mesh.guard[0],
                     j * downsample + mesh.guard[1], 0);
      f(data, result, idx_data, idx_out);
    }
  }
}

template <typename Func>
void
sample_grid_quantity3d(sim_data& data, const Grid& g, int downsample,
                       multi_array<Scalar>& result, Func f) {
  const auto& ext = g.extent();
  auto& mesh = g.mesh();
  for (int k = 0; k < ext.depth(); k++) {
    for (int j = 0; j < ext.height(); j++) {
      for (int i = 0; i < ext.width(); i++) {
        Index idx_out(i, j, k);
        Index idx_data(i * downsample + mesh.guard[0],
                       j * downsample + mesh.guard[1],
                       k * downsample + mesh.guard[2]);
        f(data, result, idx_data, idx_out);
      }
    }
  }
}

data_exporter::data_exporter(sim_environment& env, uint32_t& timestep) :
    m_env(env) {
  auto& mesh = m_env.local_grid().mesh();
  auto ext = mesh.extent_less();
  auto d = m_env.params().downsample;
  tmp_grid_data.resize(ext.width() / d, ext.height() / d,
                       ext.depth() / d);

  tmp_ptc_uint_data.resize(MAX_TRACKED);
  tmp_ptc_float_data.resize(MAX_TRACKED);
}

data_exporter::~data_exporter() {}

void
data_exporter::write_output(sim_data& data, uint32_t timestep,
                            double time) {
  if (m_fld_thread && m_fld_thread->joinable()) m_fld_thread->join();
  if (m_ptc_thread && m_ptc_thread->joinable()) m_ptc_thread->join();

  data.sync_to_host();

  // Launch a new thread to handle the field output
  m_fld_thread.reset(
      new std::thread(&Aperture::data_exporter::write_field_output,
                      this, std::ref(data), timestep, time));

  data.particles.get_tracked_ptc();
  data.photons.get_tracked_ptc();

  // Launch a new thread to handle the particle output
  m_ptc_thread.reset(
      new std::thread(&Aperture::data_exporter::write_ptc_output, this,
                      std::ref(data), timestep, time));
}

void
data_exporter::write_ptc_output(sim_data& data, uint32_t timestep,
                                double time) {
  H5File datafile(
      fmt::format("{}ptc{:04d}.h5", outputDirectory,
                  timestep / m_env.params().data_interval).c_str(),
      H5F_ACC_RDWR | H5F_ACC_TRUNC);

  add_ptc_uint_output(data, "id", [](sim_data& data, std::vector<uint32_t>& v, uint32_t n) {
                                    v[n] = data.particles.tracked_data().id[n];
                                  }, datafile);

  add_ptc_float_output(data, "p1", [](sim_data& data, std::vector<float>& v, uint32_t n) {
                                    v[n] = data.particles.tracked_data().p1[n];
                                  }, datafile);
  add_ptc_float_output(data, "p2", [](sim_data& data, std::vector<float>& v, uint32_t n) {
                                    v[n] = data.particles.tracked_data().p2[n];
                                  }, datafile);
  add_ptc_float_output(data, "p3", [](sim_data& data, std::vector<float>& v, uint32_t n) {
                                    v[n] = data.particles.tracked_data().p3[n];
                                  }, datafile);

  auto& mesh = m_env.local_grid().mesh();
  add_ptc_float_output(data, "x1", [&mesh](sim_data& data, std::vector<float>& v, uint32_t n) {
                                     auto cell = data.particles.tracked_data().cell[n];
                                     auto x = data.particles.tracked_data().x1[n];
                                     v[n] = mesh.pos(0, mesh.get_c1(cell), x);
                                  }, datafile);
  add_ptc_float_output(data, "x2", [&mesh](sim_data& data, std::vector<float>& v, uint32_t n) {
                                     auto cell = data.particles.tracked_data().cell[n];
                                     auto x = data.particles.tracked_data().x2[n];
                                     v[n] = mesh.pos(1, mesh.get_c2(cell), x);
                                  }, datafile);
  add_ptc_float_output(data, "x3", [&mesh](sim_data& data, std::vector<float>& v, uint32_t n) {
                                     auto cell = data.particles.tracked_data().cell[n];
                                     auto x = data.particles.tracked_data().x3[n];
                                     v[n] = mesh.pos(2, mesh.get_c3(cell), x);
                                  }, datafile);
  datafile.close();
}

void
data_exporter::write_field_output(sim_data& data, uint32_t timestep,
                                  double time) {
  H5File datafile(
      fmt::format("{}fld{:04d}.h5", outputDirectory,
                  timestep / m_env.params().data_interval).c_str(),
      H5F_ACC_RDWR | H5F_ACC_TRUNC);
  // add_grid_output(
  //     data, "E1",
  //     [](sim_data& data, multi_array<Scalar>& p, Index idx,
  //        Index idx_out) {
  //       p(idx_out) = data.E(0, idx) + data.Ebg(0, idx);
  //     },
  //     datafile);
  ADD_GRID_OUTPUT(
      data, "E1", { p(idx_out) = data.E(0, idx) + data.Ebg(0, idx); },
      datafile);
  ADD_GRID_OUTPUT(
      data, "E2", { p(idx_out) = data.E(1, idx) + data.Ebg(1, idx); },
      datafile);
  ADD_GRID_OUTPUT(
      data, "E3", { p(idx_out) = data.E(2, idx) + data.Ebg(2, idx); },
      datafile);
  ADD_GRID_OUTPUT(
      data, "B1", { p(idx_out) = data.B(0, idx) + data.Bbg(0, idx); },
      datafile);
  ADD_GRID_OUTPUT(
      data, "B2", { p(idx_out) = data.B(1, idx) + data.Bbg(1, idx); },
      datafile);
  ADD_GRID_OUTPUT(
      data, "B3", { p(idx_out) = data.B(2, idx) + data.Bbg(2, idx); },
      datafile);
  ADD_GRID_OUTPUT(
      data, "J1", { p(idx_out) = data.J(0, idx); }, datafile);
  ADD_GRID_OUTPUT(
      data, "J2", { p(idx_out) = data.J(1, idx); }, datafile);
  ADD_GRID_OUTPUT(
      data, "J3", { p(idx_out) = data.J(2, idx); }, datafile);
  ADD_GRID_OUTPUT(
      data, "Rho_e", { p(idx_out) = data.Rho[0](0, idx); }, datafile);
  ADD_GRID_OUTPUT(
      data, "Rho_p", { p(idx_out) = data.Rho[1](0, idx); }, datafile);
  if (data.env.params().num_species > 2) {
    ADD_GRID_OUTPUT(
        data, "Rho_i", { p(idx_out) = data.Rho[2](0, idx); }, datafile);
  }
  ADD_GRID_OUTPUT(
      data, "photon_produced",
      { p(idx_out) = data.photon_produced(idx); }, datafile);
  ADD_GRID_OUTPUT(
      data, "pair_produced", { p(idx_out) = data.pair_produced(idx); },
      datafile);
  ADD_GRID_OUTPUT(
      data, "photon_num", { p(idx_out) = data.photon_num(idx); },
      datafile);
  ADD_GRID_OUTPUT(
      data, "divE", { p(idx_out) = data.divE(idx); }, datafile);
  ADD_GRID_OUTPUT(
      data, "divB", { p(idx_out) = data.divB(idx); }, datafile);
  ADD_GRID_OUTPUT(
      data, "EdotB_avg", { p(idx_out) = data.EdotB(idx); }, datafile);
  ADD_GRID_OUTPUT(
      data, "EdotB",
      {
        p(idx_out) = data.E(0, idx) * data.B(0, idx) +
                     data.E(1, idx) * data.B(1, idx) +
                     data.E(2, idx) * data.B(2, idx);
      },
      datafile);
  add_array_output(data.ph_flux, "ph_flux", datafile);

  datafile.close();
}

template <typename Func>
void
data_exporter::add_grid_output(sim_data& data, const std::string& name,
                               Func f, H5File& file) {
  if (data.env.grid().dim() == 2) {
    sample_grid_quantity2d(data, m_env.local_grid(),
                           m_env.params().downsample, tmp_grid_data, f);

    // Actually write the temp array to hdf
    hsize_t dims[2] = {(uint32_t)tmp_grid_data.width(),
                       (uint32_t)tmp_grid_data.height()};
    DataSpace dataspace(2, dims);
    DataSet dataset =
        file.createDataSet(name, PredType::NATIVE_FLOAT, dataspace);
    dataset.write(tmp_grid_data.host_ptr(), PredType::NATIVE_FLOAT);
  }
}

void
data_exporter::add_array_output(multi_array<float>& array, const std::string& name,
                                H5File& file) {
  // Actually write the temp array to hdf
  hsize_t dims[3] = {(uint32_t)array.width(),
                     (uint32_t)array.height(),
                     (uint32_t)array.depth()};
  DataSpace dataspace(3, dims);
  DataSet dataset =
      file.createDataSet(name, PredType::NATIVE_FLOAT, dataspace);
  dataset.write(array.host_ptr(), PredType::NATIVE_FLOAT);
}

template <typename Func>
void
data_exporter::add_ptc_float_output(sim_data& data, const std::string& name, Func f,
                                    H5::H5File& file) {
  for (uint32_t n = 0; n < data.particles.tracked_number(); n++) {
    f(data, tmp_ptc_float_data, n);
  }

  hsize_t dims[1] = {data.particles.tracked_number()};
  DataSpace dataspace(1, dims);
  DataSet dataset =
      file.createDataSet(name, PredType::NATIVE_FLOAT, dataspace);
  dataset.write(tmp_ptc_float_data.data(), PredType::NATIVE_FLOAT);
}

template <typename Func>
void
data_exporter::add_ptc_uint_output(sim_data& data, const std::string& name, Func f,
                                   H5::H5File& file) {
  for (uint32_t n = 0; n < data.particles.tracked_number(); n++) {
    f(data, tmp_ptc_uint_data, n);
  }

  hsize_t dims[1] = {data.particles.tracked_number()};
  DataSpace dataspace(1, dims);
  DataSet dataset =
      file.createDataSet(name, PredType::NATIVE_UINT32, dataspace);
  dataset.write(tmp_ptc_uint_data.data(), PredType::NATIVE_UINT32);
}

}  // namespace Aperture
