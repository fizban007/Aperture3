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
  auto& params = env.params();
  auto& mesh = m_env.local_grid().mesh();
  auto out_ext = mesh.extent_less();
  auto d = m_env.params().downsample;
  for (int i = 0; i < 3; i++) {
    if (i < m_env.local_grid().dim()) {
      out_ext[i] /= d;
    }
  }
  tmp_grid_data = multi_array<float>(out_ext);
  Logger::print_info("tmp_grid_data initialized with size {}x{}x{}",
                     tmp_grid_data.width(), tmp_grid_data.height(),
                     tmp_grid_data.depth());

  // tmp_ptc_uint_data.resize(MAX_TRACKED);
  // tmp_ptc_float_data.resize(MAX_TRACKED);

  // Allocate temporary particle data
  size_t max_num =
      std::max(params.max_ptc_number, params.max_photon_number);
  tmp_ptc_data = ::operator new(max_num * 64);

  outputDirectory = env.params().data_dir;
  // make sure output directory is a directory
  if (outputDirectory.back() != '/') outputDirectory.push_back('/');
  boost::filesystem::path outPath(outputDirectory);

  boost::system::error_code returnedError;
  boost::filesystem::create_directories(outPath, returnedError);

  copy_config_file();
}

data_exporter::~data_exporter() {
  // delete[] (double*)tmp_ptc_data;
  ::operator delete(tmp_ptc_data);
}

void
data_exporter::write_grid() {
  auto& params = m_env.params();
  auto& mesh = m_env.local_grid().mesh();
  auto out_ext = tmp_grid_data.extent();
  auto downsample = params.downsample;

  std::string meshfilename = outputDirectory + "mesh.h5";
  H5File meshfile =
      hdf_create(meshfilename, H5CreateMode::trunc_parallel);

  if (m_env.local_grid().dim() == 1) {
    std::vector<float> x_array(out_ext.x);

    for (int i = 0; i < out_ext.x; i++) {
      x_array[i] = mesh.pos(0, i * downsample + mesh.guard[0], false);
    }

    meshfile.write_parallel(
        x_array.data(), out_ext.x, params.N[0] / downsample,
        mesh.offset[0] / downsample, out_ext.x, 0, "x1");
    // write_collective_array(x_array.data(), "x1",
    //                        m_env.params().N[0] / downsample, ext.x,
    //                        mesh.offset[0] / downsample, datafile);
  } else if (m_env.local_grid().dim() == 2) {
    multi_array<float> x1_array(out_ext);
    multi_array<float> x2_array(out_ext);

    for (int j = 0; j < out_ext.height(); j++) {
      for (int i = 0; i < out_ext.width(); i++) {
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
    // write_multi_array(x1_array, "x1", total_ext, offset, datafile);
    // write_multi_array(x2_array, "x2", total_ext, offset, datafile);
    meshfile.write_parallel(x1_array, total_ext, offset, out_ext,
                            Index(0, 0, 0), "x1");
    meshfile.write_parallel(x2_array, total_ext, offset, out_ext,
                            Index(0, 0, 0), "x2");
  } else if (m_env.local_grid().dim() == 3) {
    multi_array<float> x1_array(out_ext);
    multi_array<float> x2_array(out_ext);
    multi_array<float> x3_array(out_ext);

    for (int k = 0; k < out_ext.depth(); k++) {
      for (int j = 0; j < out_ext.height(); j++) {
        for (int i = 0; i < out_ext.width(); i++) {
          x1_array(i, j) =
              mesh.pos(0, i * downsample + mesh.guard[0], false);
          x2_array(i, j) =
              mesh.pos(1, j * downsample + mesh.guard[1], false);
          x3_array(i, j) =
              mesh.pos(2, k * downsample + mesh.guard[2], false);
        }
      }
    }

    Extent total_ext =
        m_env.super_grid().mesh().extent_less() / downsample;
    Index offset(mesh.offset[0] / downsample,
                 mesh.offset[1] / downsample,
                 mesh.offset[2] / downsample);
    // write_multi_array(x1_array, "x1", total_ext, offset, datafile);
    // write_multi_array(x2_array, "x2", total_ext, offset, datafile);
    // write_multi_array(x3_array, "x3", total_ext, offset, datafile);
    meshfile.write_parallel(x1_array, total_ext, offset, out_ext,
                            Index(0, 0, 0), "x1");
    meshfile.write_parallel(x2_array, total_ext, offset, out_ext,
                            Index(0, 0, 0), "x2");
    meshfile.write_parallel(x3_array, total_ext, offset, out_ext,
                            Index(0, 0, 0), "x3");
  }

  // H5Fclose(datafile);
  meshfile.close();
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
                                 const Index& offset, H5File& file) {
  file.write_parallel(array, total_ext, offset, array.extent(),
                      Index(0, 0, 0), name);
}

void
data_exporter::save_snapshot(const std::string& filename,
                             sim_data& data, uint32_t step,
                             Scalar time) {}

void
data_exporter::load_snapshot(const std::string& filename,
                             sim_data& data, uint32_t& step,
                             Scalar& time) {}

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
data_exporter::write_output(sim_data& data, uint32_t timestep,
                            double time) {
  data.copy_to_host();

  if (!m_xmf.is_open()) {
    m_xmf.open(outputDirectory + "data.xmf");
  }
  write_xmf_step_header(m_xmf_buffer, time);

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

  data.particles.get_tracked_ptc();
  data.photons.get_tracked_ptc();

  write_ptc_output(data, timestep, time);
}

void
data_exporter::write_ptc_output(sim_data& data, uint32_t timestep,
                                double time) {
  SimParams& params = m_env.params();
  auto& ptc = data.particles;
  auto& ph = data.photons;
  std::vector<uint64_t> tracked(params.num_species + 1, 0);
  std::vector<uint64_t> offset(params.num_species + 1, 0);
  std::vector<uint64_t> total(params.num_species + 1, 0);

  for (uint64_t n = 0; n < ptc.tracked_number(); n++) {
    for (int i = 0; i < params.num_species; i++) {
      if (get_ptc_type(ptc.tracked_data().flag[n]) == i)
        tracked[i] += 1;
    }
  }
  // last species is photon
  tracked[params.num_species] = ph.tracked_number();

  // Carry out an MPI scan to get the total number and local offset
  for (int i = 0; i < params.num_species + 1; i++) {
    auto status = MPI_Scan(&tracked[i], &offset[i], 1, MPI_UINT64_T,
                           MPI_SUM, m_env.world());
    offset[i] -= tracked[i];
    status = MPI_Allreduce(&tracked[i], &total[i], 1, MPI_UINT64_T,
                           MPI_SUM, m_env.world());
    // TODO: handle error here
    MPI_Helper::handle_mpi_error(status, m_env.domain_info().rank);
  }

  std::string filename =
      fmt::format("{}ptc.{:05d}.h5", outputDirectory,
                  timestep / m_env.params().data_interval);
  H5File datafile = hdf_create(filename, H5CreateMode::trunc_parallel);
  // user_write_ptc_output(data, *this, tracked, offset, total,
  // timestep,
  //                       time, datafile);

  datafile.close();
}

void
data_exporter::write_field_output(sim_data& data, uint32_t timestep,
                                  double time) {
  std::string filename =
      fmt::format("{}fld.{:05d}.h5", outputDirectory,
                  timestep / m_env.params().data_interval);
  H5File datafile = hdf_create(filename, H5CreateMode::trunc_parallel);

  user_write_field_output(data, *this, timestep, time, datafile);
  datafile.close();
}

template <typename Func>
void
data_exporter::add_grid_output(sim_data& data, const std::string& name,
                               Func f, H5File& file,
                               uint32_t timestep) {
  int downsample = m_env.params().downsample;
  if (data.env.grid().dim() == 3) {
    sample_grid_quantity3d(data, m_env.local_grid(),
                           m_env.params().downsample, tmp_grid_data, f);
  } else if (data.env.grid().dim() == 2) {
    sample_grid_quantity2d(data, m_env.local_grid(),
                           m_env.params().downsample, tmp_grid_data, f);
  } else if (data.env.grid().dim() == 1) {
    sample_grid_quantity1d(data, m_env.local_grid(),
                           m_env.params().downsample, tmp_grid_data, f);
  }
  Extent dims;
  Index offset;
  for (int i = 0; i < 3; i++) {
    dims[i] = m_env.params().N[i];
    if (dims[i] > downsample) dims[i] /= downsample;
    offset[i] = m_env.grid().mesh().offset[i] / downsample;
  }

  write_multi_array(tmp_grid_data, name, dims, offset, file);

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

template <typename T>
void
data_exporter::add_grid_output(multi_array<T>& array, Stagger stagger,
                               const std::string& name, H5File& file,
                               uint32_t timestep) {
  int downsample = m_env.params().downsample;
  auto& mesh = m_env.local_grid().mesh();
  array.downsample(downsample, tmp_grid_data,
                   Index(mesh.guard[0], mesh.guard[1], mesh.guard[2]),
                   stagger);

  Extent dims;
  Index offset;
  for (int i = 0; i < 3; i++) {
    dims[i] = m_env.params().N[i];
    if (dims[i] > downsample) dims[i] /= downsample;
    offset[i] = m_env.grid().mesh().offset[i] / downsample;
  }
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
                                Stagger stagger,
                                const std::string& name, H5File& file,
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

// template <typename Func>
// void
// data_exporter::add_ptc_float_output(sim_data& data,
//                                     const std::string& name,
//                                     uint64_t num, uint64_t total,
//                                     uint64_t offset, Func f,
//                                     H5File& file, uint32_t timestep)
//                                     {
//   // Logger::print_info("writing the {} of {} tracked particles",
//   name,
//   //                    num);
//   uint32_t num_subset = 0;
//   for (uint32_t n = 0; n < num; n++) {
//     f(data, tmp_ptc_float_data, n, num_subset);
//   }

//   write_collective_array(tmp_ptc_float_data.data(), name, total,
//                          num_subset, offset, file_id);
// }

// template <typename Func>
// void
// data_exporter::add_ptc_uint_output(sim_data& data,
//                                    const std::string& name,
//                                    uint64_t num, uint64_t total,
//                                    uint64_t offset, Func f,
//                                    H5File& file, uint32_t timestep) {
//   uint32_t num_subset = 0;
//   for (uint32_t n = 0; n < num; n++) {
//     f(data, tmp_ptc_uint_data, n, num_subset);
//   }

//   write_collective_array(tmp_ptc_uint_data.data(), name, total,
//                          num_subset, offset, file_id);
// }

}  // namespace Aperture
