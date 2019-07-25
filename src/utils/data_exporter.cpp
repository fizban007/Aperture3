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

#define ADD_GRID_OUTPUT(exporter, input, name, func, file, step)       \
  exporter.add_grid_output(input, name,                                \
                           [](sim_data & data, multi_array<float> & p, \
                              Index idx, Index idx_out) func,          \
                           file, step)

#include "utils/user_data_output.hpp"

using namespace HighFive;

namespace Aperture {

template <typename Func>
void
sample_grid_quantity1d(sim_data& data, const Grid& g, int downsample,
                       multi_array<float>& result,
                       std::vector<float>& out, Func f) {
  // const auto& ext = g.extent();
  auto& mesh = g.mesh();
  for (unsigned int i = 0; i < out.size(); i++) {
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
  const auto& ext = result.extent();
  // Logger::print_info("output ext is {}x{}", ext.width(),
  // ext.height());
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
  const auto& ext = result.extent();
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
  for (uint32_t i = 0; i < 3; i++) {
    if (i < m_env.local_grid().dim()) {
      ext[i] /= d;
    }
  }
  tmp_grid_data = multi_array<float>(ext);
  Logger::print_info("tmp_grid_data initialized with size {}x{}x{}",
                     tmp_grid_data.width(), tmp_grid_data.height(),
                     tmp_grid_data.depth());
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
  // make sure output directory is a directory
  if (outputDirectory.back() != '/') outputDirectory.push_back('/');
  boost::filesystem::path outPath(outputDirectory);

  boost::system::error_code returnedError;
  boost::filesystem::create_directories(outPath, returnedError);

  copy_config_file();
}

data_exporter::~data_exporter() {}

void
data_exporter::write_grid() {
  auto& mesh = m_env.local_grid().mesh();
  auto ext = tmp_grid_data.extent();
  auto downsample = m_env.params().downsample;
  if (m_env.local_grid().dim() == 1) {
    std::vector<float> x_array(ext.x);

    for (int i = 0; i < ext.x; i++) {
      x_array[i] = mesh.pos(0, i * downsample + mesh.guard[0], false);
    }

    std::string meshfilename = outputDirectory + "mesh.h5";
    Logger::print_info("{}", meshfilename);
    File meshfile(meshfilename,
                  File::ReadWrite | File::Create | File::Truncate);
    DataSet mesh_x1 =
        meshfile.createDataSet<float>("x1", DataSpace::From(x_array));
    mesh_x1.write(x_array);
  } else if (m_env.local_grid().dim() == 2) {
    boost::multi_array<float, 2> x1_array(
        boost::extents[ext.height()][ext.width()]);
    boost::multi_array<float, 2> x2_array(
        boost::extents[ext.height()][ext.width()]);

    for (int j = 0; j < ext.height(); j++) {
      for (int i = 0; i < ext.width(); i++) {
        if (m_env.params().coord_system == "LogSpherical") {
          float r = std::exp(
              mesh.pos(0, i * downsample + mesh.guard[0], false));
          float theta =
              mesh.pos(1, j * downsample + mesh.guard[1], false);
          x1_array[j][i] = r * std::sin(theta);
          x2_array[j][i] = r * std::cos(theta);
        } else {
          x1_array[j][i] =
              mesh.pos(0, i * downsample + mesh.guard[0], false);
          x2_array[j][i] =
              mesh.pos(1, j * downsample + mesh.guard[1], false);
        }
      }
    }

    std::string meshfilename = outputDirectory + "mesh.h5";
    Logger::print_info("{}", meshfilename);
    File meshfile(meshfilename,
                  File::ReadWrite | File::Create | File::Truncate);
    DataSet mesh_x1 =
        meshfile.createDataSet<float>("x1", DataSpace::From(x1_array));
    mesh_x1.write(x1_array);
    DataSet mesh_x2 =
        meshfile.createDataSet<float>("x2", DataSpace::From(x2_array));
    mesh_x2.write(x2_array);
  }
}

void
data_exporter::copy_config_file() {
  std::string path = outputDirectory + "config.toml";
  boost::filesystem::copy_file(
      m_env.params().conf_file, path,
      boost::filesystem::copy_option::overwrite_if_exists);
}

void
data_exporter::write_xmf_head(std::ofstream& fs) {
  fs << "<?xml version=\"1.0\" ?>" << std::endl;
  fs << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>" << std::endl;
  fs << "<Xdmf>" << std::endl;
  fs << "<Domain>" << std::endl;
  fs << "<Grid Name=\"Aperture\" GridType=\"Collection\" "
        "CollectionType=\"Temporal\" >"
     << std::endl;
}

void
data_exporter::write_xmf_step_header(std::ofstream& fs, double time) {
  // std::string dim_str;
  auto& grid = m_env.local_grid();
  // auto &mesh = grid.mesh();
  if (grid.dim() == 3) {
    m_dim_str =
        fmt::format("{} {} {}", tmp_grid_data.depth(),
                    tmp_grid_data.height(), tmp_grid_data.width());
  } else if (grid.dim() == 2) {
    m_dim_str = fmt::format("{} {}", tmp_grid_data.height(),
                            tmp_grid_data.width());
  } else if (grid.dim() == 1) {
    m_dim_str = fmt::format("{} 1", tmp_grid_data.width());
  }

  fs << "<Grid Name=\"quadmesh\" Type=\"Uniform\">" << std::endl;
  fs << "  <Time Type=\"Single\" Value=\"" << time << "\"/>"
     << std::endl;
  if (grid.dim() == 3) {
    fs << "  <Topology Type=\"3DSMesh\" NumberOfElements=\""
       << m_dim_str << "\"/>" << std::endl;
    fs << "  <Geometry GeometryType=\"X_Y_Z\">" << std::endl;
  } else if (grid.dim() == 2) {
    fs << "  <Topology Type=\"2DSMesh\" NumberOfElements=\""
       << m_dim_str << "\"/>" << std::endl;
    fs << "  <Geometry GeometryType=\"X_Y\">" << std::endl;
  } else if (grid.dim() == 1) {
    fs << "  <Topology Type=\"2DSMesh\" NumberOfElements=\""
       << m_dim_str << "\"/>" << std::endl;
    fs << "  <Geometry GeometryType=\"X_Y\">" << std::endl;
  }
  fs << "    <DataItem Dimensions=\"" << m_dim_str
     << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
     << std::endl;
  fs << "      mesh.h5:x1" << std::endl;
  fs << "    </DataItem>" << std::endl;
  if (grid.dim() >= 2) {
    fs << "    <DataItem Dimensions=\"" << m_dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << "      mesh.h5:x2" << std::endl;
    fs << "    </DataItem>" << std::endl;
  }
  if (grid.dim() >= 3) {
    fs << "    <DataItem Dimensions=\"" << m_dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << "      mesh.h5:x3" << std::endl;
    fs << "    </DataItem>" << std::endl;
  }

  fs << "  </Geometry>" << std::endl;
}

void
data_exporter::write_xmf_step_close(std::ofstream& fs) {
  fs << "</Grid>" << std::endl;
}

void
data_exporter::write_xmf_tail(std::ofstream& fs) {
  fs << "</Grid>" << std::endl;
  fs << "</Domain>" << std::endl;
  fs << "</Xdmf>" << std::endl;
}

void
data_exporter::prepare_xmf_restart(uint32_t restart_step,
                                   int data_interval, float time) {
  boost::filesystem::path xmf_file(outputDirectory + "data.xmf");
  boost::filesystem::path xmf_bak(outputDirectory + "data.xmf.bak");
  boost::filesystem::remove(xmf_bak);
  boost::filesystem::rename(xmf_file, xmf_bak);

  std::ifstream xmf_in;
  xmf_in.open(xmf_bak.c_str());

  m_xmf.open(xmf_file.c_str());

  // int n = -1;
  // int num_outputs = restart_step / data_interval;
  std::string line;
  bool in_step = false, found = false;
  std::string t_line = "  <Time Type=\"Single\" Value=\"";
  while (std::getline(xmf_in, line)) {
    if (line == "<Grid Name=\"quadmesh\" Type=\"Uniform\">") {
      // n += 1;
      in_step = true;
    }
    if (in_step && line.compare(0, t_line.length(), t_line) == 0) {
      std::string sub = line.substr(line.find_first_of("0123456789"));
      sub = sub.substr(0, sub.find_first_of("\""));
      float t = std::stof(sub);
      if (std::abs(t - time) < 1.0e-4) found = true;
    }
    m_xmf << line << std::endl;
    if (line == "</Grid>") in_step = false;
    if (found && !in_step) break;
    // if (n >= num_outputs && !in_step) break;
  }
  write_xmf_tail(m_xmf);

  xmf_in.close();
}

void
data_exporter::write_snapshot(sim_data& data, uint32_t step) {}

void
data_exporter::load_from_snapshot(sim_data& data, uint32_t step,
                                  double time) {}

void
data_exporter::write_output(sim_data& data, uint32_t timestep,
                            double time) {
  data.sync_to_host();

  if (!m_xmf.is_open()) {
    m_xmf.open(outputDirectory + "data.xmf");
  }
  if (timestep == 0) {
    write_xmf_head(m_xmf);
  } else {
    m_xmf.seekp(-26, std::ios_base::end);
  }
  write_xmf_step_header(m_xmf, time);
  // Launch a new thread to handle the field output
  // m_fld_thread.reset(
  //     new std::thread(&Aperture::data_exporter::write_field_output,
  //                     this, std::ref(data), timestep, time));
  write_field_output(data, timestep, time);

  write_xmf_step_close(m_xmf);
  write_xmf_tail(m_xmf);

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
}

void
data_exporter::write_field_output(sim_data& data, uint32_t timestep,
                                  double time) {
  File datafile(fmt::format("{}fld.{:05d}.h5", outputDirectory,
                            timestep / m_env.params().data_interval),
                File::ReadWrite | File::Create | File::Truncate);
  // MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

  user_write_field_output(data, *this, timestep, time, datafile);
}

template <typename Func>
void
data_exporter::add_grid_output(sim_data& data, const std::string& name,
                               Func f, File& file, uint32_t timestep) {
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
    // Logger::print_info("data space size {}x{}", dims[0], dims[1]);
    // Actually write the temp array to hdf
    auto dataset = file.createDataSet<float>(name, DataSpace(dims));

    std::vector<size_t> out_dim(2);
    std::vector<size_t> offsets(2);
    offsets[0] = m_env.grid().mesh().offset[1] / downsample;
    out_dim[0] = tmp_grid_data.extent()[1];
    offsets[1] = m_env.grid().mesh().offset[0] / downsample;
    out_dim[1] = tmp_grid_data.extent()[0];
    // Logger::print_info("offset is {}x{}", offsets[0], offsets[1]);
    // Logger::print_info("out_dim is {}x{}", out_dim[0], out_dim[1]);
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
    // Logger::print_info("offset is {}, dim is {}", offsets[0],
    //                    out_dim[0]);
    dataset.select(offsets, out_dim).write(m_output_1d);
  }
  m_xmf << "  <Attribute Name=\"" << name
        << "\" Center=\"Node\" AttributeType=\"Scalar\">" << std::endl;
  m_xmf << "    <DataItem Dimensions=\"" << m_dim_str
        << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
        << std::endl;
  m_xmf << fmt::format("      fld.{:05d}.h5:{}",
                       timestep / m_env.params().data_interval, name)
        << std::endl;
  m_xmf << "    </DataItem>" << std::endl;
  m_xmf << "  </Attribute>" << std::endl;
}

void
data_exporter::add_array_output(multi_array<float>& array,
                                const std::string& name, File& file) {
  // Actually write the temp array to hdf
  // hsize_t dims[3] = {(uint32_t)array.width(),
  // (uint32_t)array.height(),
  //                    (uint32_t)array.depth()};
  // DataSpace dataspace(3, dims);
  // DataSet dataset =
  //     file.createDataSet(name, PredType::NATIVE_FLOAT, dataspace);
  // dataset.write(array.host_ptr(), PredType::NATIVE_FLOAT);
}

template <typename Func>
void
data_exporter::add_ptc_float_output(sim_data& data,
                                    const std::string& name, size_t num,
                                    Func f, File& file,
                                    uint32_t timestep) {
  Logger::print_info("writing the {} of {} tracked particles", name,
                     num);
  uint32_t num_subset = 0;
  for (uint32_t n = 0; n < num; n++) {
    f(data, tmp_ptc_float_data, n, num_subset);
  }

  // TODO: Consider MPI!!!
  DataSet dataset =
      file.createDataSet<float>(name, DataSpace({num_subset}));
  dataset.write(tmp_ptc_float_data);
}

template <typename Func>
void
data_exporter::add_ptc_uint_output(sim_data& data,
                                   const std::string& name, size_t num,
                                   Func f, File& file,
                                   uint32_t timestep) {
  uint32_t num_subset = 0;
  for (uint32_t n = 0; n < num; n++) {
    f(data, tmp_ptc_uint_data, n, num_subset);
  }

  // TODO: Consider MPI!!!
  DataSet dataset =
      file.createDataSet<uint32_t>(name, DataSpace({num_subset}));
  dataset.write(tmp_ptc_uint_data);
}

}  // namespace Aperture
