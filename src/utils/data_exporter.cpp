#include "H5Cpp.h"
#include "data_exporter_impl.hpp"

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

void
data_exporter::write_output(sim_data &data, uint32_t timestep, double time) {
  data.sync_to_host();
}

void
data_exporter::write_field_output(sim_data& data, uint32_t timestep,
                            double time) {
  H5File datafile(fmt::format("{}{}{:06d}.h5", outputDirectory,
                              filePrefix, timestep)
                      .c_str(),
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

}  // namespace Aperture
