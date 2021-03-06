#include "data_exporter.h"
#include "core/constant_defs.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "sim_params.h"
#include "utils/mpi_helper.h"
#include <boost/filesystem.hpp>
#include <fmt/core.h>
#include <type_traits>
#include <vector>

// #define H5_USE_BOOST

// #include <highfive/H5DataSet.hpp>
// #include <highfive/H5DataSpace.hpp>
// #include <highfive/H5File.hpp>

#define ADD_GRID_OUTPUT(exporter, input, name, func, file, step)       \
  exporter.add_grid_output(input, name,                                \
                           [](sim_data & data, multi_array<float> & p, \
                              Index idx, Index idx_out) func,          \
                           file, step)

#include "utils/user_data_output.hpp"

namespace Aperture {

template <typename Func>
void
sample_grid_quantity1d(sim_data& data, const Grid& g, int downsample,
                       multi_array<float>& result, Func f) {
  // const auto& ext = g.extent();
  auto& mesh = g.mesh();
  for (int i = 0; i < result.extent().width(); i++) {
    Index idx_out(i, 0, 0);
    Index idx_data(i * downsample + mesh.guard[0], 0, 0);
    f(data, result, idx_data, idx_out);
  }
}

template <typename Func>
void
sample_grid_quantity2d(sim_data& data, const Grid& g, int downsample,
                       multi_array<float>& result, Func f) {
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
    }
  }
}

template <typename Func>
void
sample_grid_quantity3d(sim_data& data, const Grid& g, int downsample,
                       multi_array<float>& result, Func f) {
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
  // if (mesh.dim() == 3) {
  //   m_output_3d.resize(
  //       boost::extents[tmp_grid_data.depth()][tmp_grid_data.height()]
  //                     [tmp_grid_data.width()]);
  // } else if (mesh.dim() == 2) {
  //   m_output_2d.resize(
  //       boost::extents[tmp_grid_data.height()][tmp_grid_data.width()]);
  // } else {  // 1D
  //   m_output_1d.resize(tmp_grid_data.width());
  // }

  // tmp_ptc_uint_data.resize(MAX_TRACKED);
  // tmp_ptc_float_data.resize(MAX_TRACKED);

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

  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

  std::string meshfilename = outputDirectory + "mesh.h5";
  hid_t datafile = H5Fcreate(meshfilename.c_str(), H5F_ACC_TRUNC,
                             H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);

  if (m_env.local_grid().dim() == 1) {
    std::vector<float> x_array(ext.x);

    for (int i = 0; i < ext.x; i++) {
      x_array[i] = mesh.pos(0, i * downsample + mesh.guard[0], false);
    }

    write_collective_array(x_array.data(), "x1",
                           m_env.params().N[0] / downsample, ext.x,
                           mesh.offset[0] / downsample, datafile);
  } else if (m_env.local_grid().dim() == 2) {
    multi_array<float> x1_array(ext);
    multi_array<float> x2_array(ext);

    for (int j = 0; j < ext.height(); j++) {
      for (int i = 0; i < ext.width(); i++) {
        if (m_env.params().coord_system == "LogSpherical") {
          float r = std::exp(
              mesh.pos(0, i * downsample + mesh.guard[0], false));
          float theta =
              mesh.pos(1, j * downsample + mesh.guard[1], false);
          x1_array(i, j) = r * std::sin(theta);
          x2_array(i, j) = r * std::cos(theta);
        } else {
          x1_array(i, j) =
              mesh.pos(0, i * downsample + mesh.guard[0], false);
          x2_array(i, j) =
              mesh.pos(1, j * downsample + mesh.guard[1], false);
        }
      }
    }

    Extent total_ext =
        m_env.super_grid().mesh().extent_less() / downsample;
    total_ext.z = 1;
    Logger::print_info("total ext is {}x{}x{}", total_ext.x,
                       total_ext.y, total_ext.z);
    Index offset(mesh.offset[0] / downsample,
                 mesh.offset[1] / downsample, 0);
    Logger::print_info("offset is {}x{}x{}", offset.x, offset.y,
                       offset.z);
    write_multi_array(x1_array, "x1", total_ext, offset, datafile);
    write_multi_array(x2_array, "x2", total_ext, offset, datafile);
    // File meshfile(meshfilename,
    //               File::ReadWrite | File::Create | File::Truncate);
    // DataSet mesh_x1 =
    //     meshfile.createDataSet<float>("x1",
    //     DataSpace::From(x1_array));
    // mesh_x1.write(x1_array);
    // DataSet mesh_x2 =
    //     meshfile.createDataSet<float>("x2",
    //     DataSpace::From(x2_array));
    // mesh_x2.write(x2_array);
  } else if (m_env.local_grid().dim() == 3) {
    multi_array<float> x1_array(ext);
    multi_array<float> x2_array(ext);
    multi_array<float> x3_array(ext);

    for (int k = 0; k < ext.depth(); k++) {
      for (int j = 0; j < ext.height(); j++) {
        for (int i = 0; i < ext.width(); i++) {
          if (m_env.params().coord_system == "LogSpherical") {
            float r = std::exp(
                mesh.pos(0, i * downsample + mesh.guard[0], false));
            float theta =
                mesh.pos(1, j * downsample + mesh.guard[1], false);
            float phi =
                mesh.pos(2, k * downsample + mesh.guard[2], false);
            x1_array(i, j, k) = r * std::sin(theta) * std::cos(phi);
            x2_array(i, j, k) = r * std::sin(theta) * std::sin(phi);
            x3_array(i, j, k) = r * std::cos(theta);
          } else {
            x1_array(i, j) =
                mesh.pos(0, i * downsample + mesh.guard[0], false);
            x2_array(i, j) =
                mesh.pos(1, j * downsample + mesh.guard[1], false);
            x3_array(i, j) =
                mesh.pos(2, k * downsample + mesh.guard[2], false);
          }
        }
      }
    }

    Extent total_ext =
        m_env.super_grid().mesh().extent_less() / downsample;
    Index offset(mesh.offset[0] / downsample,
                 mesh.offset[1] / downsample,
                 mesh.offset[2] / downsample);
    write_multi_array(x1_array, "x1", total_ext, offset, datafile);
    write_multi_array(x2_array, "x2", total_ext, offset, datafile);
    write_multi_array(x3_array, "x3", total_ext, offset, datafile);
  }

  H5Fclose(datafile);
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
data_exporter::write_xmf_step_header(std::string& buffer, double time) {
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

  buffer += "<Grid Name=\"quadmesh\" Type=\"Uniform\">\n";
  buffer +=
      fmt::format("  <Time Type=\"Single\" Value=\"{}\"/>\n", time);
  if (grid.dim() == 3) {
    buffer += fmt::format(
        "  <Topology Type=\"3DSMesh\" NumberOfElements=\"{}\"/>\n",
        m_dim_str);
    buffer += "  <Geometry GeometryType=\"X_Y_Z\">\n";
  } else if (grid.dim() == 2) {
    buffer += fmt::format(
        "  <Topology Type=\"2DSMesh\" NumberOfElements=\"{}\"/>\n",
        m_dim_str);
    buffer += "  <Geometry GeometryType=\"X_Y\">\n";
  } else if (grid.dim() == 1) {
    buffer += fmt::format(
        "  <Topology Type=\"2DSMesh\" NumberOfElements=\"{}\"/>\n",
        m_dim_str);
    buffer += "  <Geometry GeometryType=\"X_Y\">\n";
  }
  buffer += fmt::format(
      "    <DataItem Dimensions=\"{}\" NumberType=\"Float\" "
      "Precision=\"4\" Format=\"HDF\">\n",
      m_dim_str);
  buffer += "      mesh.h5:x1\n";
  buffer += "    </DataItem>\n";
  if (grid.dim() >= 2) {
    buffer += fmt::format(
        "    <DataItem Dimensions=\"{}\" NumberType=\"Float\" "
        "Precision=\"4\" Format=\"HDF\">\n",
        m_dim_str);
    buffer += "      mesh.h5:x2\n";
    buffer += "    </DataItem>\n";
  }
  if (grid.dim() >= 3) {
    buffer += fmt::format(
        "    <DataItem Dimensions=\"{}\" NumberType=\"Float\" "
        "Precision=\"4\" Format=\"HDF\">\n",
        m_dim_str);
    buffer += "      mesh.h5:x3\n";
    buffer += "    </DataItem>\n";
  }

  buffer += "  </Geometry>\n";
}

void
data_exporter::write_xmf_step_close(std::ofstream& fs) {
  fs << "</Grid>" << std::endl;
}

void
data_exporter::write_xmf_step_close(std::string& buffer) {
  buffer += "</Grid>\n";
}

void
data_exporter::write_xmf_tail(std::ofstream& fs) {
  fs << "</Grid>" << std::endl;
  fs << "</Domain>" << std::endl;
  fs << "</Xdmf>" << std::endl;
}

void
data_exporter::write_xmf_tail(std::string& buffer) {
  buffer += "</Grid>\n";
  buffer += "</Domain>\n";
  buffer += "</Xdmf>\n";
}

void
data_exporter::write_multi_array(const multi_array<float>& array,
                                 const std::string& name,
                                 const Extent& total_ext,
                                 const Index& offset, hid_t file_id) {
  hsize_t total_dim[3];
  hsize_t offsets[3];
  hsize_t local_dim[3];
  for (int i = 0; i < 3; i++) {
    total_dim[2 - i] = total_ext[i];
    offsets[2 - i] = offset[i];
    local_dim[2 - i] = array.extent()[i];
  }
  Logger::print_debug("output grid {}x{}x{}", total_dim[0],
                      total_dim[1], total_dim[2]);
  // TODO: Check the order is correct
  // Logger::print_debug("env grid dim is {}", m_env.grid().dim());
  auto filespace =
      H5Screate_simple(m_env.grid().dim(), total_dim, NULL);
  auto memspace = H5Screate_simple(m_env.grid().dim(), local_dim, NULL);
  auto plist_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(plist_id, m_env.grid().dim(), local_dim);
  auto dset_id =
      H5Dcreate(file_id, name.c_str(), H5T_NATIVE_FLOAT, filespace,
                H5P_DEFAULT, plist_id, H5P_DEFAULT);
  H5Pclose(plist_id);
  H5Sclose(filespace);

  hsize_t count[3];
  hsize_t stride[3];
  for (int i = 0; i < 3; i++) {
    count[i] = 1;
    stride[i] = 1;
  }
  filespace = H5Dget_space(dset_id);
  auto status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets,
                                    stride, count, local_dim);

  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace,
                    plist_id, array.host_ptr());

  H5Dclose(dset_id);
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Pclose(plist_id);
}

template <typename T>
void
data_exporter::write_collective_array(const T* array,
                                      const std::string& name,
                                      size_t total, size_t local,
                                      size_t offset, hid_t file_id) {
  hsize_t total_dim[1] = {total};
  hsize_t offsets[1] = {offset};
  hsize_t local_dim[1] = {local};

  // Use correct H5 type according to input type
  hid_t type_id = H5T_NATIVE_CHAR;
  if (std::is_same<T, float>::value) {
    type_id = H5T_NATIVE_FLOAT;
  } else if (std::is_same<T, double>::value) {
    type_id = H5T_NATIVE_DOUBLE;
  } else if (std::is_same<T, uint32_t>::value) {
    type_id = H5T_NATIVE_UINT32;
  } else if (std::is_same<T, uint64_t>::value) {
    type_id = H5T_NATIVE_UINT64;
  }

  auto filespace = H5Screate_simple(1, total_dim, NULL);
  auto memspace = H5Screate_simple(1, local_dim, NULL);
  auto plist_id = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(plist_id, 1, local_dim);
  auto dset_id = H5Dcreate(file_id, name.c_str(), type_id, filespace,
                           H5P_DEFAULT, plist_id, H5P_DEFAULT);
  H5Pclose(plist_id);
  H5Sclose(filespace);

  hsize_t count[1] = {1};
  hsize_t stride[1] = {1};
  filespace = H5Dget_space(dset_id);
  auto status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offsets,
                                    stride, count, local_dim);

  plist_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  status = H5Dwrite(dset_id, type_id, memspace, filespace, plist_id,
                    tmp_grid_data.host_ptr());

  H5Dclose(dset_id);
  H5Sclose(filespace);
  H5Sclose(memspace);
  H5Pclose(plist_id);
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
  data.copy_to_host();

  if (!m_xmf.is_open()) {
    m_xmf.open(outputDirectory + "data.xmf");
  }
  // write_xmf_step_header(m_xmf, time);
  write_xmf_step_header(m_xmf_buffer, time);
  // Launch a new thread to handle the field output
  // m_fld_thread.reset(
  //     new std::thread(&Aperture::data_exporter::write_field_output,
  //                     this, std::ref(data), timestep, time));
  write_field_output(data, timestep, time);

  write_xmf_step_close(m_xmf_buffer);
  write_xmf_tail(m_xmf_buffer);

  if (timestep == 0) {
    write_xmf_head(m_xmf);
  } else {
    m_xmf.seekp(-26, std::ios_base::end);
  }
  m_xmf << m_xmf_buffer;
  m_xmf_buffer = "";
}

void
data_exporter::write_ptc_output(sim_data& data, uint32_t timestep,
                                double time) {}

void
data_exporter::write_field_output(sim_data& data, uint32_t timestep,
                                  double time) {
  // File datafile(fmt::format("{}fld.{:05d}.h5", outputDirectory,
  //                           timestep / m_env.params().data_interval),
  //               File::ReadWrite | File::Create | File::Truncate);
  std::string filename =
      fmt::format("{}fld.{:05d}.h5", outputDirectory,
                  timestep / m_env.params().data_interval);
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

  hid_t datafile =
      H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
  H5Pclose(plist_id);
  // MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

  user_write_field_output(data, *this, timestep, time, datafile);
  H5Fclose(datafile);
}

template <typename Func>
void
data_exporter::add_grid_output(sim_data& data, const std::string& name,
                               Func f, hid_t file_id,
                               uint32_t timestep) {
  int downsample = m_env.params().downsample;
  if (data.env.grid().dim() == 3) {
    sample_grid_quantity3d(data, m_env.local_grid(),
                           m_env.params().downsample, tmp_grid_data, f);

  } else if (data.env.grid().dim() == 2) {
    sample_grid_quantity2d(data, m_env.local_grid(),
                           m_env.params().downsample, tmp_grid_data, f);

    // std::vector<size_t> dims(2);
    // dims[0] = m_env.params().N[1] / downsample;
    // dims[1] = m_env.params().N[0] / downsample;

  } else if (data.env.grid().dim() == 1) {
    sample_grid_quantity1d(data, m_env.local_grid(),
                           m_env.params().downsample, tmp_grid_data, f);

    // std::vector<size_t> dims(1);
    // dims[0] = m_env.params().N[0] / downsample;
  }
  Extent dims;
  Index offset;
  for (int i = 0; i < 3; i++) {
    dims[i] = m_env.params().N[i];
    if (dims[i] > downsample) dims[i] /= downsample;
    offset[i] = m_env.grid().mesh().offset[i] / downsample;
  }

  write_multi_array(tmp_grid_data, name, dims, offset, file_id);
  // m_xmf << "  <Attribute Name=\"" << name
  //       << "\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //       std::endl;
  // m_xmf << "    <DataItem Dimensions=\"" << m_dim_str
  //       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //       << std::endl;
  // m_xmf << fmt::format("      fld.{:05d}.h5:{}",
  //                      timestep / m_env.params().data_interval, name)
  //       << std::endl;
  // m_xmf << "    </DataItem>" << std::endl;
  // m_xmf << "  </Attribute>" << std::endl;
  m_xmf_buffer += fmt::format(
      "  <Attribute Name=\"{}\" Center=\"Node\" "
      "AttributeType=\"Scalar\">\n",
      name);
  m_xmf_buffer += fmt::format(
      "    <DataItem Dimensions=\"{}\" NumberType=\"Float\" "
      "Precision=\"4\" Format=\"HDF\">\n",
      m_dim_str);
  m_xmf_buffer +=
      fmt::format("      fld.{:05d}.h5:{}\n",
                  timestep / m_env.params().data_interval, name);
  m_xmf_buffer += "    </DataItem>\n";
  m_xmf_buffer += "  </Attribute>\n";
}

void
data_exporter::add_array_output(multi_array<float>& array,
                                const std::string& name, hid_t file_id,
                                uint32_t timestep) {
  // Actually write the temp array to hdf
  // std::vector<size_t> dims(1);
  // dims[0] = (uint32_t)array.size();
  // DataSet dataset = file.createDataSet<float>(name, DataSpace(dims));
  // dataset.write(array.host_ptr());
}

// void
// data_exporter::add_ptc_output(sim_data& data, int species,
//                               hid_t file_id, uint32_t timestep) {

// }

}  // namespace Aperture
