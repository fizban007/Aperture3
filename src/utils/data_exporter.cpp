#include "data_exporter.h"
#include "core/constant_defs.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "sim_params.h"
#include <boost/filesystem.hpp>
#include <fmt/core.h>

#define H5_USE_BOOST

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

#define ADD_GRID_OUTPUT(exporter, input, name, func, file)              \
  exporter.add_grid_output(input, name,                                 \
                           [](sim_data & data, multi_array<float> & p,  \
                              Index idx, Index idx_out) func,           \
                           file)

#include "utils/user_data_output.hpp"

using namespace HighFive;

namespace Aperture {

template <typename Func>
void
sample_grid_quantity1d(sim_data& data, const Grid& g, int downsample,
                       multi_array<float>& result,
                       std::vector<float>& out, Func f) {
  const auto& ext = g.extent();
  auto& mesh = g.mesh();
  for (int i = 0; i < ext.width(); i++) {
    Index idx_out(i, 0, 0);
    Index idx_data(i * downsample + mesh.guard[0], 0, 0);
    f(data, result, idx_data, idx_out);

    out[i] = result(i, 0, 0);
  }
}

template <typename Func>
void
sample_grid_quantity2d(sim_data& data, const Grid& g, int downsample,
                       multi_array<float>& result,
                       boost::multi_array<float, 2>& out, Func f) {
  const auto& ext = g.extent();
  auto& mesh = g.mesh();
  for (int j = 0; j < ext.height(); j++) {
    for (int i = 0; i < ext.width(); i++) {
      Index idx_out(i, j, 0);
      Index idx_data(i * downsample + mesh.guard[0],
                     j * downsample + mesh.guard[1], 0);
      f(data, result, idx_data, idx_out);

      out[j][i] = result(i, j, 0);
    }
  }
}

template <typename Func>
void
sample_grid_quantity3d(sim_data& data, const Grid& g, int downsample,
                       multi_array<float>& result,
                       boost::multi_array<float, 3>& out, Func f) {
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

        out[k][j][i] = result(i, j, k);
      }
    }
  }
}

data_exporter::data_exporter(sim_environment& env, uint32_t& timestep)
    : m_env(env) {
  auto& mesh = m_env.local_grid().mesh();
  auto ext = mesh.extent_less();
  auto d = m_env.params().downsample;
  tmp_grid_data = multi_array<float>(ext.width() / d, ext.height() / d,
                                     ext.depth() / d);
  if (mesh.dim() == 3) {
    m_output_3d.resize(
        boost::extents[tmp_grid_data.depth()][tmp_grid_data.height()]
                      [tmp_grid_data.width()]);
  } else if (mesh.dim() == 2) {
    m_output_2d.resize(
        boost::extents[tmp_grid_data.height()][tmp_grid_data.width()]);
  } else {  // 1D
    m_output_1d.resize(tmp_grid_data.width());
  }

  tmp_ptc_uint_data.resize(MAX_TRACKED);
  tmp_ptc_float_data.resize(MAX_TRACKED);

  outputDirectory = env.params().data_dir;
  boost::filesystem::path outPath(outputDirectory);

  boost::system::error_code returnedError;
  boost::filesystem::create_directories(outPath, returnedError);

  std::string path = outputDirectory + "config.toml";
  boost::filesystem::copy_file(
      env.params().conf_file, path,
      boost::filesystem::copy_option::overwrite_if_exists);
}

data_exporter::~data_exporter() {}

void
data_exporter::write_output(sim_data& data, uint32_t timestep,
                            double time) {
  data.sync_to_host();

  // Launch a new thread to handle the field output
  // m_fld_thread.reset(
  //     new std::thread(&Aperture::data_exporter::write_field_output,
  //                     this, std::ref(data), timestep, time));
  write_field_output(data, timestep, time);

  data.particles.get_tracked_ptc();
  data.photons.get_tracked_ptc();

  // Launch a new thread to handle the particle output
  // m_ptc_thread.reset(
  //     new std::thread(&Aperture::data_exporter::write_ptc_output,
  //     this,
  //                     std::ref(data), timestep, time));
  write_ptc_output(data, timestep, time);
}

void
data_exporter::write_ptc_output(sim_data& data, uint32_t timestep,
                                double time) {
  File datafile(fmt::format("{}ptc.{:05d}.h5", outputDirectory,
                            timestep / m_env.params().data_interval),
                File::ReadWrite | File::Create | File::Truncate);
                // MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));
  user_write_ptc_output(data, *this, timestep, time, datafile);
  // add_ptc_uint_output(
  //     data, "id",
  //     [](sim_data& data, std::vector<uint32_t>& v, uint32_t n) {
  //       v[n] = data.particles.tracked_data().id[n];
  //     },
  //     datafile);

  // add_ptc_float_output(
  //     data, "p1",
  //     [](sim_data& data, std::vector<float>& v, uint32_t n) {
  //       v[n] = data.particles.tracked_data().p1[n];
  //     },
  //     datafile);
  // add_ptc_float_output(
  //     data, "p2",
  //     [](sim_data& data, std::vector<float>& v, uint32_t n) {
  //       v[n] = data.particles.tracked_data().p2[n];
  //     },
  //     datafile);
  // add_ptc_float_output(
  //     data, "p3",
  //     [](sim_data& data, std::vector<float>& v, uint32_t n) {
  //       v[n] = data.particles.tracked_data().p3[n];
  //     },
  //     datafile);

  // auto& mesh = m_env.local_grid().mesh();
  // add_ptc_float_output(
  //     data, "x1",
  //     [&mesh](sim_data& data, std::vector<float>& v, uint32_t n) {
  //       auto cell = data.particles.tracked_data().cell[n];
  //       auto x = data.particles.tracked_data().x1[n];
  //       v[n] = mesh.pos(0, mesh.get_c1(cell), x);
  //     },
  //     datafile);
  // add_ptc_float_output(
  //     data, "x2",
  //     [&mesh](sim_data& data, std::vector<float>& v, uint32_t n) {
  //       auto cell = data.particles.tracked_data().cell[n];
  //       auto x = data.particles.tracked_data().x2[n];
  //       v[n] = mesh.pos(1, mesh.get_c2(cell), x);
  //     },
  //     datafile);
  // add_ptc_float_output(
  //     data, "x3",
  //     [&mesh](sim_data& data, std::vector<float>& v, uint32_t n) {
  //       auto cell = data.particles.tracked_data().cell[n];
  //       auto x = data.particles.tracked_data().x3[n];
  //       v[n] = mesh.pos(2, mesh.get_c3(cell), x);
  //     },
  //     datafile);
  // datafile.close();
}

void
data_exporter::write_field_output(sim_data& data, uint32_t timestep,
                                  double time) {
  File datafile(fmt::format("{}fld.{:05d}.h5", outputDirectory,
                            timestep / m_env.params().data_interval)
                    .c_str(),
                File::ReadWrite | File::Create | File::Truncate,
                MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

  user_write_field_output(data, *this, timestep, time, datafile);
  // add_grid_output(
  //     data, "E1",
  //     [](sim_data& data, multi_array<Scalar>& p, Index idx,
  //        Index idx_out) {
  //       p(idx_out) = data.E(0, idx) + data.Ebg(0, idx);
  //     },
  //     datafile);
  // ADD_GRID_OUTPUT(
  //     data, "E1",
  //     {
  //       p(idx_out) = 0.25 * (data.E(0, idx) +
  //                            data.E(0, idx.x, idx.y + 1, idx.z) +
  //                            data.E(0, idx.x, idx.y, idx.z + 1) +
  //                            data.E(0, idx.x, idx.y + 1, idx.z + 1));
  //     },
  //     datafile);
  // ADD_GRID_OUTPUT(
  //     data, "E2",
  //     {
  //       p(idx_out) = 0.25 * (data.E(1, idx) +
  //                            data.E(1, idx.x + 1, idx.y, idx.z) +
  //                            data.E(1, idx.x, idx.y, idx.z + 1) +
  //                            data.E(1, idx.x + 1, idx.y, idx.z + 1));
  //     },
  //     datafile);
  // ADD_GRID_OUTPUT(
  //     data, "E3",
  //     {
  //       p(idx_out) = 0.25 * (data.E(2, idx) +
  //                            data.E(2, idx.x + 1, idx.y, idx.z) +
  //                            data.E(2, idx.x, idx.y + 1, idx.z) +
  //                            data.E(2, idx.x + 1, idx.y + 1, idx.z));
  //     },
  //     datafile);
  // ADD_GRID_OUTPUT(
  //     data, "B1",
  //     {
  //       p(idx_out) =
  //           0.5 * (data.B(0, idx) + data.B(0, idx.x - 1, idx.y, idx.z));
  //     },
  //     datafile);
  // ADD_GRID_OUTPUT(
  //     data, "B2",
  //     {
  //       p(idx_out) =
  //           0.5 * (data.B(1, idx) + data.B(1, idx.x, idx.y - 1, idx.z));
  //     },
  //     datafile);
  // ADD_GRID_OUTPUT(
  //     data, "B3",
  //     {
  //       p(idx_out) =
  //           0.5 * (data.B(2, idx) + data.B(2, idx.x, idx.y, idx.z - 1));
  //     },
  //     datafile);
  // ADD_GRID_OUTPUT(
  //     data, "J1", { p(idx_out) = data.J(0, idx); }, datafile);
  // ADD_GRID_OUTPUT(
  //     data, "J2", { p(idx_out) = data.J(1, idx); }, datafile);
  // ADD_GRID_OUTPUT(
  //     data, "J3", { p(idx_out) = data.J(2, idx); }, datafile);
  // ADD_GRID_OUTPUT(
  //     data, "Rho_e", { p(idx_out) = data.Rho[0](0, idx); }, datafile);
  // ADD_GRID_OUTPUT(
  //     data, "Rho_p", { p(idx_out) = data.Rho[1](0, idx); }, datafile);
  // if (data.env.params().num_species > 2) {
  //   ADD_GRID_OUTPUT(
  //       data, "Rho_i", { p(idx_out) = data.Rho[2](0, idx); }, datafile);
  // }
  // ADD_GRID_OUTPUT(
  //     data, "photon_produced",
  //     { p(idx_out) = data.photon_produced(idx); }, datafile);
  // ADD_GRID_OUTPUT(
  //     data, "pair_produced", { p(idx_out) = data.pair_produced(idx); },
  //     datafile);
  // ADD_GRID_OUTPUT(
  //     data, "photon_num", { p(idx_out) = data.photon_num(idx); },
  //     datafile);
  // ADD_GRID_OUTPUT(
  //     data, "divE", { p(idx_out) = data.divE(idx); }, datafile);
  // ADD_GRID_OUTPUT(
  //     data, "divB", { p(idx_out) = data.divB(idx); }, datafile);
  // ADD_GRID_OUTPUT(
  //     data, "EdotB_avg", { p(idx_out) = data.EdotB(idx); }, datafile);
  // add_array_output(data.ph_flux, "ph_flux", datafile);

  // datafile.close();
}

template <typename Func>
void
data_exporter::add_grid_output(sim_data& data, const std::string& name,
                               Func f, File& file) {
  int downsample = m_env.params().downsample;
  if (data.env.grid().dim() == 3) {
    sample_grid_quantity3d(data, m_env.local_grid(),
                           m_env.params().downsample, tmp_grid_data,
                           m_output_3d, f);

    std::vector<size_t> dims(3);
    for (int i = 0; i < 3; i++) {
      dims[i] = m_env.params().N[2 - i];
      if (dims[i] > (size_t)downsample) dims[i] /= downsample;
    }
    // Actually write the temp array to hdf
    DataSet dataset = file.createDataSet<float>(name, DataSpace(dims));

    std::vector<size_t> out_dim(3);
    std::vector<size_t> offsets(3);
    for (int i = 0; i < 3; i++) {
      offsets[i] = m_env.grid().mesh().offset[2 - i] / downsample;
      out_dim[i] = tmp_grid_data.extent()[2 - i];
    }
    dataset.select(offsets, out_dim).write(m_output_2d);
  } else if (data.env.grid().dim() == 2) {
    sample_grid_quantity2d(data, m_env.local_grid(),
                           m_env.params().downsample, tmp_grid_data,
                           m_output_2d, f);

    std::vector<size_t> dims(2);
    dims[0] = m_env.params().N[1] / downsample;
    dims[1] = m_env.params().N[0] / downsample;
    // Actually write the temp array to hdf
    DataSet dataset = file.createDataSet<float>(name, DataSpace(dims));

    std::vector<size_t> out_dim(2);
    std::vector<size_t> offsets(2);
    offsets[0] = m_env.grid().mesh().offset[1] / downsample;
    out_dim[0] = tmp_grid_data.extent()[1];
    offsets[1] = m_env.grid().mesh().offset[0] / downsample;
    out_dim[1] = tmp_grid_data.extent()[0];
    dataset.select(offsets, out_dim).write(m_output_2d);
  } else if (data.env.grid().dim() == 1) {
    sample_grid_quantity1d(data, m_env.local_grid(),
                           m_env.params().downsample, tmp_grid_data,
                           m_output_1d, f);

    std::vector<size_t> dims(1);
    dims[0] = m_env.params().N[0] / downsample;
    // Actually write the temp array to hdf
    DataSet dataset = file.createDataSet<float>(name, DataSpace(dims));

    std::vector<size_t> out_dim(1);
    std::vector<size_t> offsets(1);
    offsets[0] = m_env.grid().mesh().offset[0] / downsample;
    out_dim[0] = tmp_grid_data.extent()[0];
    Logger::print_info("offset is {}, dim is {}", offsets[0], out_dim[0]);
    dataset.select(offsets, out_dim).write(m_output_1d);
  }
}

void
data_exporter::add_array_output(multi_array<float>& array,
                                const std::string& name, File& file) {
  // Actually write the temp array to hdf
  // hsize_t dims[3] = {(uint32_t)array.width(), (uint32_t)array.height(),
  //                    (uint32_t)array.depth()};
  // DataSpace dataspace(3, dims);
  // DataSet dataset =
  //     file.createDataSet(name, PredType::NATIVE_FLOAT, dataspace);
  // dataset.write(array.host_ptr(), PredType::NATIVE_FLOAT);
}

template <typename Func>
void
data_exporter::add_ptc_float_output(sim_data& data,
                                    const std::string& name, size_t num, Func f,
                                    File& file) {
  Logger::print_info("writing the {} of {} tracked particles", name, num);
  for (uint32_t n = 0; n < num; n++) {
    f(data, tmp_ptc_float_data, n);
  }

  // TODO: Consider MPI!!!
  DataSet dataset =
      file.createDataSet<float>(name, DataSpace({num}));
  dataset.write(tmp_ptc_float_data);
}

template <typename Func>
void
data_exporter::add_ptc_uint_output(sim_data& data,
                                   const std::string& name, size_t num, Func f,
                                   File& file) {
  for (uint32_t n = 0; n < num; n++) {
    f(data, tmp_ptc_uint_data, n);
  }

  // TODO: Consider MPI!!!
  DataSet dataset =
      file.createDataSet<uint32_t>(name, DataSpace({num}));
  dataset.write(tmp_ptc_uint_data);
}

}  // namespace Aperture
