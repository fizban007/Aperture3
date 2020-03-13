#include "data_exporter.h"
#include "core/constant_defs.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "sim_params.h"
#include "utils/mpi_helper.h"
#include <boost/filesystem.hpp>
#include <fmt/ostream.h>
#include <type_traits>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "visit_struct/visit_struct.hpp"

#define ADD_GRID_OUTPUT(exporter, input, name, func, file, step)       \
  exporter.add_grid_output(input, name,                                \
                           [](sim_data & data, multi_array<float> & p, \
                              Index idx, Index idx_out) func,          \
                           file, step)

// #include "utils/user_data_output.hpp"

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
  m_out_ext = m_env.super_grid().mesh().extent_less();
  auto d = m_env.params().downsample;
  for (int i = 0; i < 3; i++) {
    if (i < m_env.local_grid().dim()) {
      out_ext[i] /= d;
      m_out_ext[i] /= d;
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
  tmp_ptc_data = ::operator new(max_num * 8);

  outputDirectory = env.params().data_dir;
  // make sure output directory is a directory
  if (outputDirectory.back() != '/') outputDirectory.push_back('/');
  boost::filesystem::path outPath(outputDirectory);

  boost::system::error_code returnedError;
  boost::filesystem::create_directories(outPath, returnedError);

  copy_config_file();
  write_grid();
}

data_exporter::~data_exporter() { ::operator delete(tmp_ptc_data); }

void
data_exporter::write_grid() {
  auto& params = m_env.params();
  auto& mesh = m_env.local_grid().mesh();
  auto out_ext = tmp_grid_data.extent();
  auto downsample = params.downsample;

  std::string meshfilename = outputDirectory + "grid.h5";
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
  if (m_env.domain_info().rank != 0) return;
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
  if (m_env.domain_info().rank != 0) return;
  // std::string dim_str;
  auto& grid = m_env.super_grid();
  auto &mesh = grid.mesh();
  auto ext = mesh.extent_less();
  if (grid.dim() == 3) {
    m_dim_str =
        fmt::format("{} {} {}", m_out_ext.depth(),
                    m_out_ext.height(), m_out_ext.width());
  } else if (grid.dim() == 2) {
    m_dim_str = fmt::format("{} {}", m_out_ext.height(),
                            m_out_ext.width());
  } else if (grid.dim() == 1) {
    m_dim_str = fmt::format("{} 1", m_out_ext.width());
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
  fs << "      grid.h5:x1" << std::endl;
  fs << "    </DataItem>" << std::endl;
  if (grid.dim() >= 2) {
    fs << "    <DataItem Dimensions=\"" << m_dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << "      grid.h5:x2" << std::endl;
    fs << "    </DataItem>" << std::endl;
  }
  if (grid.dim() >= 3) {
    fs << "    <DataItem Dimensions=\"" << m_dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << "      grid.h5:x3" << std::endl;
    fs << "    </DataItem>" << std::endl;
  }

  fs << "  </Geometry>" << std::endl;
}

void
data_exporter::write_xmf_step_header(std::string& buffer, double time) {
  if (m_env.domain_info().rank != 0) return;
  // std::string dim_str;
  // auto& grid = m_env.local_grid();
  auto& grid = m_env.super_grid();
  auto &mesh = grid.mesh();
  auto ext = mesh.extent_less();
  // auto &mesh = grid.mesh();
  if (grid.dim() == 3) {
    m_dim_str =
        fmt::format("{} {} {}", m_out_ext.depth(),
                    m_out_ext.height(), m_out_ext.width());
  } else if (grid.dim() == 2) {
    m_dim_str = fmt::format("{} {}", m_out_ext.height(),
                            m_out_ext.width());
  } else if (grid.dim() == 1) {
    m_dim_str = fmt::format("{} 1", m_out_ext.width());
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
  buffer += "      grid.h5:x1\n";
  buffer += "    </DataItem>\n";
  if (grid.dim() >= 2) {
    buffer += fmt::format(
        "    <DataItem Dimensions=\"{}\" NumberType=\"Float\" "
        "Precision=\"4\" Format=\"HDF\">\n",
        m_dim_str);
    buffer += "      grid.h5:x2\n";
    buffer += "    </DataItem>\n";
  }
  if (grid.dim() >= 3) {
    buffer += fmt::format(
        "    <DataItem Dimensions=\"{}\" NumberType=\"Float\" "
        "Precision=\"4\" Format=\"HDF\">\n",
        m_dim_str);
    buffer += "      grid.h5:x3\n";
    buffer += "    </DataItem>\n";
  }

  buffer += "  </Geometry>\n";
}

void
data_exporter::write_xmf_step_close(std::ofstream& fs) {
  if (m_env.domain_info().rank != 0) return;
  fs << "</Grid>" << std::endl;
}

void
data_exporter::write_xmf_step_close(std::string& buffer) {
  if (m_env.domain_info().rank != 0) return;
  buffer += "</Grid>\n";
}

void
data_exporter::write_xmf_tail(std::ofstream& fs) {
  if (m_env.domain_info().rank != 0) return;
  fs << "</Grid>" << std::endl;
  fs << "</Domain>" << std::endl;
  fs << "</Xdmf>" << std::endl;
}

void
data_exporter::write_xmf_tail(std::string& buffer) {
  if (m_env.domain_info().rank != 0) return;
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
                             Scalar time) {
  // Sync to host regardless of cpu or gpu
  data.copy_to_host();

  // Open snapshot file for writing
  auto datafile = hdf_create(filename, H5CreateMode::trunc_parallel);

  int rank = m_env.domain_info().rank;
  int num_ranks = m_env.domain_info().size;
  auto& params = m_env.params();
  auto& mesh = m_env.mesh();
  Extent ext_total, ext;
  Index idx_dst, idx_src;
  for (int i = 0; i < mesh.dim(); i++) {
    ext_total[i] = params.N[i] + 2 * params.guard[i];
    ext[i] = mesh.reduced_dim(i);
    idx_dst[i] = mesh.offset[i];
    idx_src[i] = 0;
    if (idx_dst[i] > 0) {
      idx_dst[i] += mesh.guard[i];
      idx_src[i] += mesh.guard[i];
    }
    if (m_env.domain_info().neighbor_left[i] == MPI_PROC_NULL) {
      ext[i] += mesh.guard[i];
    }
    if (m_env.domain_info().neighbor_right[i] == MPI_PROC_NULL) {
      ext[i] += mesh.guard[i];
    }
  }

  Logger::print_debug("ext_total is {}, idx_dst is {}, ext is {}", ext_total, idx_dst, ext);

  // Write to snapshot file
  datafile.write_parallel(data.E.data(0), ext_total, idx_dst, ext,
                          idx_src, "Ex");
  datafile.write_parallel(data.E.data(1), ext_total, idx_dst, ext,
                          idx_src, "Ey");
  datafile.write_parallel(data.E.data(2), ext_total, idx_dst, ext,
                          idx_src, "Ez");
  datafile.write_parallel(data.B.data(0), ext_total, idx_dst, ext,
                          idx_src, "Bx");
  datafile.write_parallel(data.B.data(1), ext_total, idx_dst, ext,
                          idx_src, "By");
  datafile.write_parallel(data.B.data(2), ext_total, idx_dst, ext,
                          idx_src, "Bz");
  datafile.write_parallel(data.Ebg.data(0), ext_total, idx_dst, ext,
                          idx_src, "E0x");
  datafile.write_parallel(data.Ebg.data(1), ext_total, idx_dst, ext,
                          idx_src, "E0y");
  datafile.write_parallel(data.Ebg.data(2), ext_total, idx_dst, ext,
                          idx_src, "E0z");
  datafile.write_parallel(data.Bbg.data(0), ext_total, idx_dst, ext,
                          idx_src, "B0x");
  datafile.write_parallel(data.Bbg.data(1), ext_total, idx_dst, ext,
                          idx_src, "B0y");
  datafile.write_parallel(data.Bbg.data(2), ext_total, idx_dst, ext,
                          idx_src, "B0z");
#ifdef USE_CUDA
  size_t rand_array_size = 1024 * 512 * data.rand_state_size;
  datafile.write_parallel((char*)data.d_rand_states, rand_array_size,
                          rand_array_size * num_ranks,
                          rand_array_size * rank, rand_array_size, 0,
                          "rand_states");
#endif

  // No need to write diagnostics, or derived quantities like current,
  // rho, density, etc.
  // write the number of particles in each rank into the hdf5 file
  uint64_t ptc_num = data.particles.number();
  uint64_t ph_num = data.photons.number();
  datafile.write_parallel(&ptc_num, 1, m_env.domain_info().size,
                          m_env.domain_info().rank, 1, 0, "ptc_num");
  datafile.write_parallel(&ph_num, 1, m_env.domain_info().size,
                          m_env.domain_info().rank, 1, 0, "ph_num");

  datafile.write(step, "step");
  datafile.write(time, "time");
  datafile.write(params.data_interval, "data_interval");
  datafile.write(num_ranks, "num_ranks");
  datafile.write(m_output_num, "output_num");

  add_ptc_output(data.particles.data(), data.particles.number(),
                 datafile, "ptc_");
  add_ptc_output(data.photons.data(), data.photons.number(), datafile,
                 "ph_");

  datafile.close();
}

void
data_exporter::load_snapshot(const std::string& filename,
                             sim_data& data, uint32_t& step,
                             Scalar& time) {
  // Check whether filename exists
  if (!boost::filesystem::exists(filename)) {
    Logger::print_info(
        "Can't find restart file, proceeding without loading it!");
    return;
  }

  // Open the snapshot file for reading
  H5File datafile(filename, H5OpenMode::read_parallel);

  int rank = m_env.domain_info().rank;
  int num_ranks = m_env.domain_info().size;
  int restart_ranks = datafile.read_scalar<int>("num_ranks");

  if (num_ranks != restart_ranks) {
    Logger::print_err(
        "Restarting with different rank configuration is not allowed!");
    exit(1);
  }

  auto& params = m_env.params();
  auto& grid = m_env.grid();
  auto& mesh = m_env.mesh();
  Extent ext;
  Index idx_dst, idx_src;
  for (int i = 0; i < mesh.dim(); i++) {
    // ext_total[i] = params.N[i] + 2 * params.guard[i];
    ext[i] = mesh.reduced_dim(i);
    idx_src[i] = mesh.offset[i];
    idx_dst[i] = 0;
    if (idx_src[i] > 0) {
      idx_src[i] += mesh.guard[i];
      idx_dst[i] += mesh.guard[i];
    }
    if (m_env.domain_info().neighbor_left[i] == MPI_PROC_NULL) {
      ext[i] += mesh.guard[i];
    }
    if (m_env.domain_info().neighbor_right[i] == MPI_PROC_NULL) {
      ext[i] += mesh.guard[i];
    }
  }

  Logger::print_debug("idx_dst is {}, ext is {}, idx_src is {}", idx_dst, ext, idx_src);

  datafile.read_subset(data.E.data(0), "Ex", idx_src, ext, idx_dst);
  datafile.read_subset(data.E.data(1), "Ey", idx_src, ext, idx_dst);
  datafile.read_subset(data.E.data(2), "Ez", idx_src, ext, idx_dst);
  datafile.read_subset(data.B.data(0), "Bx", idx_src, ext, idx_dst);
  datafile.read_subset(data.B.data(1), "By", idx_src, ext, idx_dst);
  datafile.read_subset(data.B.data(2), "Bz", idx_src, ext, idx_dst);
  datafile.read_subset(data.Bbg.data(0), "B0x", idx_src, ext, idx_dst);
  datafile.read_subset(data.Bbg.data(1), "B0y", idx_src, ext, idx_dst);
  datafile.read_subset(data.Bbg.data(2), "B0z", idx_src, ext, idx_dst);
  datafile.read_subset(data.Ebg.data(0), "E0x", idx_src, ext, idx_dst);
  datafile.read_subset(data.Ebg.data(1), "E0y", idx_src, ext, idx_dst);
  datafile.read_subset(data.Ebg.data(2), "E0z", idx_src, ext, idx_dst);

#ifdef USE_CUDA
  size_t rand_array_size = 1024 * 512 * data.rand_state_size;
  datafile.read_subset((char*)data.d_rand_states, rand_array_size,
                          "rand_states", rand_array_size * rank,
                          rand_array_size, 0);
#endif

  step = datafile.read_scalar<uint32_t>("step");
  time = datafile.read_scalar<Scalar>("time");
  m_output_num = datafile.read_scalar<int>("output_num");

  data.copy_to_device();
  data.init_bg_fields();

  m_env.send_guard_cells(data.E);
  m_env.send_guard_cells(data.B);
  m_env.send_guard_cells(data.Ebg);
  m_env.send_guard_cells(data.Bbg);

  // Read particle numbers
  uint64_t ptc_num, ph_num;
  datafile.read_subset(&ptc_num, 1, "ptc_num", rank, 1, 0);
  datafile.read_subset(&ph_num, 1, "ph_num", rank, 1, 0);

  read_ptc_output(data.particles.data(), ptc_num, datafile, "ptc_");
  read_ptc_output(data.photons.data(), ph_num, datafile, "ph_");

  Logger::print_debug("Read {} particles from restart", ptc_num);
  Logger::print_debug("Read {} photons from restart", ph_num);
  data.particles.set_num(ptc_num);
  data.photons.set_num(ph_num);
  data.particles.clear_guard_cells(grid);
  data.photons.clear_guard_cells(grid);

  auto data_interval = datafile.read_scalar<size_t>("data_interval");
  prepare_xmf_restart(step, data_interval, time);

  datafile.close();

  // data.particles.sort_by_cell(grid);
  // data.sort_particles();
}

void
data_exporter::prepare_xmf_restart(uint32_t restart_step,
                                   int data_interval, float time) {
  if (m_env.domain_info().rank != 0) return;
  boost::filesystem::path xmf_file(outputDirectory + "data.xmf");
  boost::filesystem::path xmf_bak(outputDirectory + "data.xmf.bak");
  boost::filesystem::remove(xmf_bak);
  if (boost::filesystem::exists(xmf_file)) {
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
  } else {
    write_xmf_head(m_xmf);
  }

}

void
data_exporter::write_output(sim_data& data, uint32_t timestep,
                            double time) {
  data.copy_to_host();

  if (!m_xmf.is_open() && m_env.domain_info().rank == 0) {
    m_xmf.open(outputDirectory + "data.xmf");
  }
  write_xmf_step_header(m_xmf_buffer, time);

  write_field_output(data, timestep, time);

  if (m_env.domain_info().rank == 0) {
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

  write_ptc_output(data, timestep, time);
  m_output_num += 1;
}

void
data_exporter::write_ptc_output(sim_data& data, uint32_t timestep,
                                double time) {
  data.particles.get_tracked_ptc();
  data.photons.get_tracked_ptc();

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
  // bool is_empty = true;
  for (int i = 0; i < params.num_species + 1; i++) {
    m_env.get_total_num_offset(tracked[i], total[i], offset[i]);
    // if (total[i] > 0) is_empty = false;
  }

  // Skip if there is nothing to output
  // if (is_empty) return;

  std::string filename =
      fmt::format("{}ptc.{:05d}.h5", outputDirectory,
                  // timestep / m_env.params().data_interval);
                  m_output_num);
  H5File datafile = hdf_create(filename, H5CreateMode::trunc_parallel);

  auto& mesh = m_env.mesh();
  for (int i = 0; i <= params.num_species; i++) {
    add_tracked_ptc_output<uint64_t>(
        data, i, "id", total[i], offset[i],
        [&mesh](auto& data, uint64_t n, uint64_t nsb, uint64_t* p) {
          p[nsb] = data.id[n];
        },
        datafile);
    add_tracked_ptc_output<float>(
        data, i, "x1", total[i], offset[i],
        [&mesh](auto& data, uint64_t n, uint64_t nsb, float* p) {
          auto cell = data.cell[n];
          auto x = data.x1[n];
          p[nsb] = mesh.pos(0, mesh.get_c1(cell), x);
        },
        datafile);
    add_tracked_ptc_output<float>(
        data, i, "x2", total[i], offset[i],
        [&mesh](auto& data, uint64_t n, uint64_t nsb, float* p) {
          auto cell = data.cell[n];
          auto x = data.x2[n];
          p[nsb] = mesh.pos(1, mesh.get_c2(cell), x);
        },
        datafile);
    add_tracked_ptc_output<float>(
        data, i, "x3", total[i], offset[i],
        [&mesh](auto& data, uint64_t n, uint64_t nsb, float* p) {
          auto cell = data.cell[n];
          auto x = data.x3[n];
          p[nsb] = mesh.pos(2, mesh.get_c3(cell), x);
        },
        datafile);
    add_tracked_ptc_output<float>(
        data, i, "p1", total[i], offset[i],
        [&mesh](auto& data, uint64_t n, uint64_t nsb, float* p) {
          p[nsb] = data.p1[n];
        },
        datafile);
    add_tracked_ptc_output<float>(
        data, i, "p2", total[i], offset[i],
        [&mesh](auto& data, uint64_t n, uint64_t nsb, float* p) {
          p[nsb] = data.p2[n];
        },
        datafile);
    add_tracked_ptc_output<float>(
        data, i, "p3", total[i], offset[i],
        [&mesh](auto& data, uint64_t n, uint64_t nsb, float* p) {
          p[nsb] = data.p3[n];
        },
        datafile);
  }

  datafile.close();
}

void
data_exporter::write_field_output(sim_data& data, uint32_t timestep,
                                  double time) {
  std::string filename =
      fmt::format("{}fld.{:05d}.h5", outputDirectory,
                  // timestep / m_env.params().data_interval);
                  m_output_num);
  H5File datafile = hdf_create(filename, H5CreateMode::trunc_parallel);

  // user_write_field_output(data, *this, timestep, time, datafile);
  add_grid_output(data.E.data(0), data.E.stagger(0), "E1", datafile,
                  timestep);
  add_grid_output(data.E.data(1), data.E.stagger(1), "E2", datafile,
                  timestep);
  add_grid_output(data.E.data(2), data.E.stagger(2), "E3", datafile,
                  timestep);
  add_grid_output(data.B.data(0), data.B.stagger(0), "B1", datafile,
                  timestep);
  add_grid_output(data.B.data(1), data.B.stagger(1), "B2", datafile,
                  timestep);
  add_grid_output(data.B.data(2), data.B.stagger(2), "B3", datafile,
                  timestep);
  add_grid_output(data.J.data(0), data.J.stagger(0), "J1", datafile,
                  timestep);
  add_grid_output(data.J.data(1), data.J.stagger(1), "J2", datafile,
                  timestep);
  add_grid_output(data.J.data(2), data.J.stagger(2), "J3", datafile,
                  timestep);
  add_grid_output(data.Rho[0].data(), data.Rho[0].stagger(), "Rho_e",
                  datafile, timestep);
  add_grid_output(data.Rho[1].data(), data.Rho[1].stagger(), "Rho_p",
                  datafile, timestep);
  if (data.env.params().num_species > 2) {
    add_grid_output(data.Rho[2].data(), data.Rho[2].stagger(), "Rho_i",
                    datafile, timestep);
  }
  add_grid_output(data.divE.data(), data.divE.stagger(), "divE",
                  datafile, timestep);
  add_grid_output(data.divB.data(), data.divB.stagger(), "divB",
                  datafile, timestep);
  add_grid_output(data.photon_produced.data(), data.photon_produced.stagger(), "photon_produced",
                  datafile, timestep);
  add_grid_output(data.pair_produced.data(), data.pair_produced.stagger(), "pair_produced",
                  datafile, timestep);

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

  if (m_env.domain_info().rank == 0) {
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
  if (m_env.domain_info().rank == 0) {
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

  file.write_parallel(tmp_grid_data, dims, offset,
                      tmp_grid_data.extent(), Index(0, 0, 0), name);
}

template <typename Ptc>
void
data_exporter::add_ptc_output(Ptc& data, size_t num, H5File& file,
                              const std::string& prefix) {
  uint64_t offset, total;
  m_env.get_total_num_offset(num, total, offset);

  void* data_ptr;

  Logger::print_debug("num is {}, total is {}, offset is {}", num, total, offset);
#ifdef USE_CUDA
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, data.cell);
  bool is_device = (attributes.type == cudaMemoryTypeDevice);

  visit_struct::for_each(data, [&](const char* name, auto& ptr) {
    typedef typename std::remove_reference<decltype(*ptr)>::type x_type;
    if (is_device) {
      cudaMemcpy(tmp_ptc_data, ptr, num * sizeof(x_type),
                 cudaMemcpyDeviceToHost);
      data_ptr = tmp_ptc_data;
    } else {
      data_ptr = (void*)ptr;
    }
    file.write_parallel((x_type*)data_ptr, num, total, offset, num, 0,
                        prefix + std::string(name));
  });
#else
  visit_struct::for_each(data, [&](const char* name, auto& ptr) {
    typedef typename std::remove_reference<decltype(*ptr)>::type x_type;
    data_ptr = (void*)ptr;
    file.write_parallel((x_type*)data_ptr, num, total, offset, num, 0,
                        prefix + std::string(name));
  });
#endif
}

template <typename Ptc>
void
data_exporter::read_ptc_output(Ptc& data, size_t num, H5File& file,
                               const std::string& prefix) {

  uint64_t offset, total;
  m_env.get_total_num_offset(num, total, offset);

  void* data_ptr;

#ifdef USE_CUDA
  cudaPointerAttributes attributes;
  cudaPointerGetAttributes(&attributes, data.cell);
  bool is_device = (attributes.type == cudaMemoryTypeDevice);

  visit_struct::for_each(data, [&](const char* name, auto& ptr) {
    typedef typename std::remove_reference<decltype(*ptr)>::type x_type;
    if (is_device) {
      data_ptr = tmp_ptc_data;
    } else {
      data_ptr = (void*)ptr;
    }
    file.read_subset((x_type*)data_ptr, num, prefix + std::string(name),
                     offset, num, 0);
    if (is_device) {
      cudaMemcpy(ptr, tmp_ptc_data, num * sizeof(x_type),
                 cudaMemcpyHostToDevice);
    }
  });
#else
  visit_struct::for_each(data, [&](const char* name, auto& ptr) {
    typedef typename std::remove_reference<decltype(*ptr)>::type x_type;
    data_ptr = (void*)ptr;
    file.read_subset((x_type*)data_ptr, num, prefix + std::string(name),
                     offset, num, 0);
  });
#endif
}

template <typename T, typename Func>
void
data_exporter::add_tracked_ptc_output(sim_data& data, int sp,
                                      const std::string& name,
                                      uint64_t total, uint64_t offset,
                                      Func f, H5File& file) {
  uint64_t n_subset = 0;
  if (sp < m_env.params().num_species) {
    // Not photons
    for (uint64_t n = 0; n < data.particles.tracked_number(); n++) {
      if (get_ptc_type(data.particles.tracked_data().flag[n]) == sp) {
        f(data.particles.tracked_data(), n, n_subset, (T*)tmp_ptc_data);
        n_subset += 1;
      }
    }
  } else if (sp == m_env.params().num_species) {
    // Process photons
    for (uint64_t n = 0; n < data.photons.tracked_number(); n++) {
      f(data.photons.tracked_data(), n, n_subset, (T*)tmp_ptc_data);
      n_subset += 1;
    }
  }

  uint64_t sb_offset, sb_total;
  m_env.get_total_num_offset(n_subset, sb_total, sb_offset);
  // Logger::print_debug("n_subset is {}, sb_total is {}, sb_offset is {}", n_subset,
  //                     sb_total, sb_offset);

  // if (sb_total > 0) {
  file.write_parallel(
      (T*)tmp_ptc_data, n_subset, sb_total, sb_offset, n_subset, 0,
      fmt::format("{}_{}", particle_type_name(sp), name));
  // }
}

}  // namespace Aperture
