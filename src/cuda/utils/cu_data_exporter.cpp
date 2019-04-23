#include <cuda_runtime.h>
#include "cuda/utils/cu_data_exporter.h"
#include "core/constant_defs.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/cu_sim_data1d.h"
#include "cuda/core/sim_environment_dev.h"
#include "sim_params.h"
#include "utils/hdf_exporter_impl.hpp"
#include "utils/type_name.h"
#include <omp.h>

#define H5_USE_BOOST

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <time.h>

#include "visit_struct/visit_struct.hpp"

using namespace HighFive;

namespace Aperture {

template class hdf_exporter<cu_data_exporter>;

cu_data_exporter::cu_data_exporter(SimParams &params,
                                   uint32_t &timestep)
    : hdf_exporter(params, timestep) {}

cu_data_exporter::~cu_data_exporter() {}

void
cu_data_exporter::add_cu_field_output(const std::string &name,
                                      const std::string &type,
                                      int num_components,
                                      std::vector<field_base *> &field,
                                      int dim, bool sync) {
  auto &mesh = grid->mesh();
  if (dim == 3) {
    cu_fieldoutput<3> tempData;
    tempData.name = name;
    tempData.type = type;
    tempData.field = field;
    tempData.sync = sync;
    tempData.f.resize(num_components);
    for (int i = 0; i < num_components; i++) {
      tempData.f[i].resize(
          boost::extents[mesh.dims[2]][mesh.dims[1]][mesh.dims[0]]);
    }
    m_fields_3d.push_back(std::move(tempData));
  } else if (dim == 2) {
    cu_fieldoutput<2> tempData;
    tempData.name = name;
    tempData.type = type;
    tempData.field = field;
    tempData.sync = sync;
    tempData.f.resize(num_components);
    for (int i = 0; i < num_components; i++) {
      tempData.f[i].resize(boost::extents[mesh.dims[1]][mesh.dims[0]]);
    }
    m_fields_2d.push_back(std::move(tempData));
  } else if (dim == 1) {
    cu_fieldoutput<1> tempData;
    tempData.name = name;
    tempData.type = type;
    tempData.field = field;
    tempData.sync = sync;
    tempData.f.resize(num_components);
    for (int i = 0; i < num_components; i++) {
      tempData.f[i].resize(boost::extents[mesh.dims[0]]);
    }
    m_fields_1d.push_back(std::move(tempData));
  }
}

template <typename T>
void
cu_data_exporter::add_field(const std::string &name,
                            std::vector<cu_scalar_field<T>> &field,
                            bool sync) {
  std::vector<field_base *> field_ptr(field.size());
  for (unsigned int i = 0; i < field.size(); i++) {
    field_ptr[i] = &field[i];
    // Logger::print_info("grid is {}x{}",
    // field_ptr[i]->grid().extent().x,
    // field_ptr[i]->grid().extent().y);
  }
  add_cu_field_output(name, TypeName<T>::Get(), 1, field_ptr,
                      grid->dim(), sync);
}

template <typename T>
void
cu_data_exporter::add_field(const std::string &name,
                            std::vector<cu_vector_field<T>> &field,
                            bool sync) {
  std::vector<field_base *> field_ptr(field.size());
  for (unsigned int i = 0; i < field.size(); i++) {
    field_ptr[i] = &field[i];
  }
  add_cu_field_output(name, TypeName<T>::Get(), VECTOR_DIM, field_ptr,
                      grid->dim(), sync);
  // add_field_output(name, TypeName<T>::Get(), VECTOR_DIM, &field,
  //                  grid->dim(), sync);
}

void
cu_data_exporter::set_mesh(cu_sim_data &data) {
  m_submesh.resize(data.dev_map.size());
  m_submesh_out.resize(data.dev_map.size());
  for (unsigned int n = 0; n < m_submesh.size(); n++) {
    m_submesh[n] = data.grid[n]->mesh();
    m_submesh_out[n] = data.grid[n]->mesh();
    for (int i = 0; i < m_submesh[n].dim(); i++) {
      m_submesh_out[n].dims[i] =
          m_submesh[n].guard[i] * 2 +
          m_submesh[n].reduced_dim(i) / data.env.params().downsample;
      m_submesh_out[n].delta[i] *= data.env.params().downsample;
      m_submesh_out[n].inv_delta[i] /= data.env.params().downsample;
      m_submesh_out[n].offset[i] /= data.env.params().downsample;
    }
  }
}

// void
// cu_data_exporter::write_snapshot(cu_sim_environment &env,
//                                  cu_sim_data &data, uint32_t
//                                  timestep) {
//   File snapshotfile(
//       // fmt::format("{}snapshot{:06d}.h5", outputDirectory,
//       timestep) fmt::format("{}snapshot.h5",
//       outputDirectory).c_str(), File::ReadWrite | File::Create |
//       File::Truncate);

//   // Write background fields from environment
//   // size_t grid_size = data.E.grid().size();
//   size_t grid_size = data.E.data(0).size();
//   data.Ebg.sync_to_host();
//   DataSet data_bg_E1 =
//       snapshotfile.createDataSet<Scalar>("bg_E1",
//       DataSpace(grid_size));
//   data_bg_E1.write((char *)data.Ebg.data(0).data());
//   DataSet data_bg_E2 =
//       snapshotfile.createDataSet<Scalar>("bg_E2",
//       DataSpace(grid_size));
//   data_bg_E2.write((char *)data.Ebg.data(1).data());
//   DataSet data_bg_E3 =
//       snapshotfile.createDataSet<Scalar>("bg_E3",
//       DataSpace(grid_size));
//   data_bg_E3.write((char *)data.Ebg.data(2).data());
//   data.Bbg.sync_to_host();
//   DataSet data_bg_B1 =
//       snapshotfile.createDataSet<Scalar>("bg_B1",
//       DataSpace(grid_size));
//   data_bg_B1.write((char *)data.Bbg.data(0).data());
//   DataSet data_bg_B2 =
//       snapshotfile.createDataSet<Scalar>("bg_B2",
//       DataSpace(grid_size));
//   data_bg_B2.write((char *)data.Bbg.data(1).data());
//   DataSet data_bg_B3 =
//       snapshotfile.createDataSet<Scalar>("bg_B3",
//       DataSpace(grid_size));
//   data_bg_B3.write((char *)data.Bbg.data(2).data());

//   // Write sim data
//   // Write field values
//   data.E.sync_to_host();
//   DataSet data_E1 =
//       snapshotfile.createDataSet<Scalar>("E1", DataSpace(grid_size));
//   data_E1.write((char *)data.E.data(0).data());
//   DataSet data_E2 =
//       snapshotfile.createDataSet<Scalar>("E2", DataSpace(grid_size));
//   data_E2.write((char *)data.E.data(1).data());
//   DataSet data_E3 =
//       snapshotfile.createDataSet<Scalar>("E3", DataSpace(grid_size));
//   data_E3.write((char *)data.E.data(2).data());
//   data.B.sync_to_host();
//   DataSet data_B1 =
//       snapshotfile.createDataSet<Scalar>("B1", DataSpace(grid_size));
//   data_B1.write((char *)data.B.data(0).data());
//   DataSet data_B2 =
//       snapshotfile.createDataSet<Scalar>("B2", DataSpace(grid_size));
//   data_B2.write((char *)data.B.data(1).data());
//   DataSet data_B3 =
//       snapshotfile.createDataSet<Scalar>("B3", DataSpace(grid_size));
//   data_B3.write((char *)data.B.data(2).data());
//   data.J.sync_to_host();
//   DataSet data_J1 =
//       snapshotfile.createDataSet<Scalar>("J1", DataSpace(grid_size));
//   data_J1.write((char *)data.J.data(0).data());
//   DataSet data_J2 =
//       snapshotfile.createDataSet<Scalar>("J2", DataSpace(grid_size));
//   data_J2.write((char *)data.J.data(1).data());
//   DataSet data_J3 =
//       snapshotfile.createDataSet<Scalar>("J3", DataSpace(grid_size));
//   data_J3.write((char *)data.J.data(2).data());

//   for (int i = 0; i < data.num_species; i++) {
//     data.Rho[i].sync_to_host();
//     DataSet data_Rho = snapshotfile.createDataSet<Scalar>(
//         fmt::format("Rho{}", i), DataSpace(grid_size));
//     data_Rho.write((char *)data.Rho[i].data().data());
//   }
//   DataSet data_devId = snapshotfile.createDataSet<int>(
//       "devId", DataSpace::From(data.devId));
//   data_devId.write(data.devId);

//   // Write particle data
//   size_t ptcNum = data.particles.number();
//   DataSet data_ptcNum = snapshotfile.createDataSet<size_t>(
//       "ptcNum", DataSpace::From(ptcNum));
//   data_ptcNum.write(ptcNum);
//   Logger::print_info("Writing {} particles to snapshot", ptcNum);

//   size_t phNum = data.photons.number();
//   DataSet data_phNum = snapshotfile.createDataSet<size_t>(
//       "phNum", DataSpace::From(phNum));
//   data_phNum.write(phNum);
//   Logger::print_info("Writing {} photons to snapshot", phNum);

//   std::vector<double> buffer(std::max(ptcNum, phNum));
//   visit_struct::for_each(
//       data.particles.data(),
//       [&snapshotfile, &buffer, &ptcNum](const char *name, auto &x) {
//         typedef
//             typename std::remove_reference<decltype(*x)>::type
//             x_type;
//         DataSet ptc_data = snapshotfile.createDataSet<x_type>(
//             fmt::format("ptc_{}", name), DataSpace(ptcNum));
//         cudaMemcpy(buffer.data(), x, ptcNum * sizeof(x_type),
//                    cudaMemcpyDeviceToHost);
//         ptc_data.write(reinterpret_cast<x_type *>(buffer.data()));
//       });
//   visit_struct::for_each(
//       data.photons.data(),
//       [&snapshotfile, &buffer, &phNum](const char *name, auto &x) {
//         typedef
//             typename std::remove_reference<decltype(*x)>::type
//             x_type;
//         DataSet ph_data = snapshotfile.createDataSet<x_type>(
//             fmt::format("ph_{}", name), DataSpace(phNum));
//         cudaMemcpy(buffer.data(), x, phNum * sizeof(x_type),
//                    cudaMemcpyDeviceToHost);
//         ph_data.write(reinterpret_cast<x_type *>(buffer.data()));
//       });

//   // Write current simulation timestep and other info
//   DataSet data_timestep = snapshotfile.createDataSet<uint32_t>(
//       "timestep", DataSpace::From(timestep));
//   data_timestep.write(timestep);
// }

// void
// cu_data_exporter::load_from_snapshot(cu_sim_environment &env,
//                                      cu_sim_data &data,
//                                      uint32_t &timestep) {
//   File snapshotfile(
//       // fmt::format("{}snapshot{:06d}.h5", outputDirectory,
//       timestep) fmt::format("{}snapshot.h5",
//       outputDirectory).c_str(), File::ReadOnly);

//   // size_t grid_size = data.E.grid().size();
//   size_t ptcNum, phNum;
//   int devId;

//   // Read the scalars first
//   DataSet data_timestep = snapshotfile.getDataSet("timestep");
//   data_timestep.read(timestep);
//   DataSet data_ptcNum = snapshotfile.getDataSet("ptcNum");
//   data_ptcNum.read(ptcNum);
//   DataSet data_phNum = snapshotfile.getDataSet("phNum");
//   data_phNum.read(phNum);
//   DataSet data_devId = snapshotfile.getDataSet("devId");
//   data_devId.read(devId);

//   // Read particle data
//   std::vector<double> buffer(std::max(ptcNum, phNum));
//   data.particles.set_num(ptcNum);
//   data.photons.set_num(phNum);

//   visit_struct::for_each(
//       data.particles.data(),
//       [&snapshotfile, &buffer, &ptcNum](const char *name, auto &x) {
//         typedef
//             typename std::remove_reference<decltype(*x)>::type
//             x_type;
//         DataSet ptc_data =
//             snapshotfile.getDataSet(fmt::format("ptc_{}", name));
//         ptc_data.read(reinterpret_cast<x_type *>(buffer.data()));
//         cudaMemcpy(x, buffer.data(), ptcNum * sizeof(x_type),
//                    cudaMemcpyHostToDevice);
//       });
//   visit_struct::for_each(
//       data.photons.data(),
//       [&snapshotfile, &buffer, &phNum](const char *name, auto &x) {
//         typedef
//             typename std::remove_reference<decltype(*x)>::type
//             x_type;
//         DataSet ph_data =
//             snapshotfile.getDataSet(fmt::format("ph_{}", name));
//         ph_data.read(reinterpret_cast<x_type *>(buffer.data()));
//         cudaMemcpy(x, buffer.data(), phNum * sizeof(x_type),
//                    cudaMemcpyHostToDevice);
//       });

//   // Read field data
//   DataSet data_bg_B1 = snapshotfile.getDataSet("bg_B1");
//   data_bg_B1.read((char *)data.Bbg.data(0).data());
//   DataSet data_bg_B2 = snapshotfile.getDataSet("bg_B2");
//   data_bg_B2.read((char *)data.Bbg.data(1).data());
//   DataSet data_bg_B3 = snapshotfile.getDataSet("bg_B3");
//   data_bg_B3.read((char *)data.Bbg.data(2).data());
//   DataSet data_bg_E1 = snapshotfile.getDataSet("bg_E1");
//   data_bg_E1.read((char *)data.Ebg.data(0).data());
//   DataSet data_bg_E2 = snapshotfile.getDataSet("bg_E2");
//   data_bg_E2.read((char *)data.Ebg.data(1).data());
//   DataSet data_bg_E3 = snapshotfile.getDataSet("bg_E3");
//   data_bg_E3.read((char *)data.Ebg.data(2).data());

//   data.Bbg.sync_to_device();
//   data.Ebg.sync_to_device();

//   DataSet data_B1 = snapshotfile.getDataSet("B1");
//   data_B1.read((char *)data.B.data(0).data());
//   DataSet data_B2 = snapshotfile.getDataSet("B2");
//   data_B2.read((char *)data.B.data(1).data());
//   DataSet data_B3 = snapshotfile.getDataSet("B3");
//   data_B3.read((char *)data.B.data(2).data());
//   DataSet data_E1 = snapshotfile.getDataSet("E1");
//   data_E1.read((char *)data.E.data(0).data());
//   DataSet data_E2 = snapshotfile.getDataSet("E2");
//   data_E2.read((char *)data.E.data(1).data());
//   DataSet data_E3 = snapshotfile.getDataSet("E3");
//   data_E3.read((char *)data.E.data(2).data());
//   DataSet data_J1 = snapshotfile.getDataSet("J1");
//   data_J1.read((char *)data.J.data(0).data());
//   DataSet data_J2 = snapshotfile.getDataSet("J2");
//   data_J2.read((char *)data.J.data(1).data());
//   DataSet data_J3 = snapshotfile.getDataSet("J3");
//   data_J3.read((char *)data.J.data(2).data());
//   data.B.sync_to_device();
//   data.E.sync_to_device();
//   data.J.sync_to_device();

//   for (int i = 0; i < data.num_species; i++) {
//     DataSet data_rho = snapshotfile.getDataSet(fmt::format("Rho{}",
//     i)); data_rho.read((char *)data.Rho[i].data().data());
//     data.Rho[i].sync_to_device();
//   }
// }

template <typename T>
void
cu_data_exporter::interpolate_field_values(fieldoutput<1> &field,
                                           int components, const T &t) {
  if (components == 1) {
    auto fptr = dynamic_cast<cu_scalar_field<T> *>(field.field);
    if (field.sync) fptr->sync_to_host();
    auto &mesh = fptr->grid().mesh();
    // #pragma omp simd
    for (int i = 0; i < mesh.reduced_dim(0); i += downsample_factor) {
      field.f[0][i / downsample_factor + mesh.guard[0]] = 0.0;
      for (int n1 = 0; n1 < downsample_factor; n1++) {
        field.f[0][i / downsample_factor + mesh.guard[0]] +=
            (*fptr)(i + n1 + mesh.guard[0]) / downsample_factor;
      }
      // if (field.name == "J" && i > 230 && i < 250) {
      //   Logger::print_info(
      //       "J1 at {} is {}", i,
      //       field.f[0][i / downsample_factor + mesh.guard[0]]);
      // }
    }
  } else if (components == 3) {
    auto fptr = dynamic_cast<cu_vector_field<T> *>(field.field);
    if (field.sync) fptr->sync_to_host();
    auto &mesh = fptr->grid().mesh();
    // #pragma omp simd
    for (int i = 0; i < mesh.reduced_dim(0); i += downsample_factor) {
      field.f[0][i / downsample_factor + mesh.guard[0]] = 0.0;
      // field.f[1][i / downsample_factor + mesh.guard[0]] = 0.0;
      // field.f[2][i / downsample_factor + mesh.guard[0]] = 0.0;
      for (int n1 = 0; n1 < downsample_factor; n1++) {
        field.f[0][i / downsample_factor + mesh.guard[0]] +=
            (*fptr)(0, i + n1 + mesh.guard[0]) / downsample_factor;
      }
      // if (field.name == "J" && i > 330 && i < 370) {
      //   Logger::print_info(
      //       "J1 at {} is {}", i,
      //       field.f[0][i / downsample_factor + mesh.guard[0]]);
      // }
    }
  }
}

template <typename T>
void
cu_data_exporter::interpolate_field_values(fieldoutput<2> &field,
                                           int components, const T &t) {
  if (components == 1) {
    auto fptr = dynamic_cast<cu_scalar_field<T> *>(field.field);
    if (field.sync) fptr->sync_to_host();
    auto &mesh = fptr->grid().mesh();
#pragma omp simd collapse(2)
    for (int j = 0; j < mesh.reduced_dim(1); j += downsample_factor) {
      for (int i = 0; i < mesh.reduced_dim(0); i += downsample_factor) {
        field.f[0][j / downsample_factor + mesh.guard[1]]
               [i / downsample_factor + mesh.guard[0]] = 0.0;
        for (int n2 = 0; n2 < downsample_factor; n2++) {
          for (int n1 = 0; n1 < downsample_factor; n1++) {
            field.f[0][j / downsample_factor + mesh.guard[1]]
                   [i / downsample_factor + mesh.guard[0]] +=
                (*fptr)(i + n1 + mesh.guard[0],
                        j + n2 + mesh.guard[1]) /
                square(downsample_factor);
          }
        }
      }
      // for (int i = 0; i < mesh.reduced_dim(0); i +=
      // downsample_factor) {
      //   field.f[0][mesh.guard[1] - 1]
      //          [i / downsample_factor + mesh.guard[0]] =
      //       (*fptr)(i + mesh.guard[0], mesh.guard[1] - 1);
      // }
    }
  } else if (components == 3) {
    auto fptr = dynamic_cast<cu_vector_field<T> *>(field.field);
    if (field.sync) fptr->sync_to_host();
    auto &mesh = fptr->grid().mesh();
#pragma omp simd collapse(2)
    for (int j = 0; j < mesh.reduced_dim(1); j += downsample_factor) {
      for (int i = 0; i < mesh.reduced_dim(0); i += downsample_factor) {
        field.f[0][j / downsample_factor + mesh.guard[1]]
               [i / downsample_factor + mesh.guard[0]] = 0.0;
        field.f[1][j / downsample_factor + mesh.guard[1]]
               [i / downsample_factor + mesh.guard[0]] = 0.0;
        field.f[2][j / downsample_factor + mesh.guard[1]]
               [i / downsample_factor + mesh.guard[0]] = 0.0;
        for (int n2 = 0; n2 < downsample_factor; n2++) {
          for (int n1 = 0; n1 < downsample_factor; n1++) {
            field.f[0][j / downsample_factor + mesh.guard[1]]
                   [i / downsample_factor + mesh.guard[0]] +=
                (*fptr)(0, i + n1 + mesh.guard[0],
                        j + n2 + mesh.guard[1]) /
                square(downsample_factor);
            // std::cout << vf.f1[j / downsample_factor +
            // mesh.guard[1]
            //     [i / downsample_factor + mesh.guard[0]] <<
            // std::endl;
            field.f[1][j / downsample_factor + mesh.guard[1]]
                   [i / downsample_factor + mesh.guard[0]] +=
                (*fptr)(1, i + n1 + mesh.guard[0],
                        j + n2 + mesh.guard[1]) /
                square(downsample_factor);

            field.f[2][j / downsample_factor + mesh.guard[1]]
                   [i / downsample_factor + mesh.guard[0]] +=
                (*fptr)(2, i + n1 + mesh.guard[0],
                        j + n2 + mesh.guard[1]) /
                square(downsample_factor);
          }
        }
        assert(!isnan(field.f[0][j / downsample_factor + mesh.guard[1]]
                             [i / downsample_factor + mesh.guard[0]]));
        assert(!isnan(field.f[1][j / downsample_factor + mesh.guard[1]]
                             [i / downsample_factor + mesh.guard[0]]));
        assert(!isnan(field.f[2][j / downsample_factor + mesh.guard[1]]
                             [i / downsample_factor + mesh.guard[0]]));
      }
      // for (int i = 0; i < mesh.reduced_dim(0); i +=
      // downsample_factor) {
      //   field.f[0][mesh.guard[1] - 1]
      //          [i / downsample_factor + mesh.guard[0]] =
      //       (*fptr)(0, i + mesh.guard[0], mesh.guard[1] - 1);
      //   field.f[1][mesh.guard[1] - 1]
      //          [i / downsample_factor + mesh.guard[0]] =
      //       (*fptr)(1, i + mesh.guard[0], mesh.guard[1] - 1);
      //   field.f[2][mesh.guard[1] - 1]
      //          [i / downsample_factor + mesh.guard[0]] =
      //       (*fptr)(2, i + mesh.guard[0], mesh.guard[1] - 1);
      // }
    }
  }
}

template <typename T>
void
cu_data_exporter::interpolate_field_values(fieldoutput<3> &field,
                                           int components, const T &t) {
}

void
cu_data_exporter::write_particles(cu_sim_data &data, uint32_t step,
                                  double time) {
  // auto &mesh = orig_grid->mesh();
  // for (auto &ptcoutput : dbPtcData1d) {
  //   Particles_1D *ptc = dynamic_cast<Particles_1D *>(ptcoutput.ptc);
  //   if (ptc != nullptr) {
  //     Logger::print_info("Writing tracked particles");
  //     File datafile(fmt::format("{}{}{:06d}.h5", outputDirectory,
  //                               filePrefix, step)
  //                       .c_str(),
  //                   File::ReadWrite);
  //     for (int sp = 0; sp < m_params.num_species; sp++) {
  //       unsigned int idx = 0;
  //       std::string name_x = NameStr((ParticleType)sp) + "_x";
  //       std::string name_p = NameStr((ParticleType)sp) + "_p";
  //       for (Index_t n = 0; n < ptc->number(); n++) {
  //         if (m_ptc_cell[n] != MAX_CELL &&
  //             // ptc->check_flag(n, ParticleFlag::tracked) &&
  //             check_bit(m_ptc_flag[n], ParticleFlag::tracked) &&
  //             // ptc->check_type(n) == (ParticleType)sp &&
  //             get_ptc_type(m_ptc_flag[n]) == sp && idx < MAX_TRACKED)
  //             {
  //           Scalar x = mesh.pos(0, m_ptc_cell[n], m_ptc_x1[n]);
  //           ptcoutput.x[idx] = x;
  //           ptcoutput.p[idx] = m_ptc_p1[n];
  //           idx += 1;
  //         }
  //       }
  //       DataSet data_x =
  //           datafile.createDataSet<float>(name_x, DataSpace{idx});
  //       data_x.write(ptcoutput.x);
  //       DataSet data_p =
  //           datafile.createDataSet<float>(name_p, DataSpace{idx});
  //       data_p.write(ptcoutput.p);

  //       // Logger::print_info("Written {} tracked particles", idx);
  //     }
  //   }
  // }
  Logger::print_info("Writing tracked particles");
  File datafile(
      fmt::format("{}{}{:06d}.h5", outputDirectory, filePrefix, step)
          .c_str(),
      File::ReadWrite);
  for (auto &ptcoutput : m_ptcdata) {
    for (int sp = 0; sp < m_params.num_species; sp++) {
      unsigned int idx = 0;
      for (unsigned int dev = 0; dev < m_submesh.size(); dev++) {
        Particles *ptc = dynamic_cast<Particles *>(ptcoutput.ptc[dev]);
        if (ptc != nullptr) {
          auto &data = ptc->data();
          auto &mesh = m_submesh[dev];
          // std::string name_x = NameStr((ParticleType)sp) + "_x";
          // std::string name_p = NameStr((ParticleType)sp) + "_p";
          for (Index_t n = 0; n < ptc->number(); n++) {
            if (data.cell[n] != MAX_CELL &&
                check_bit(data.flag[n], ParticleFlag::tracked) &&
                get_ptc_type(data.flag[n]) == sp && idx < MAX_TRACKED) {
              int c1 = mesh.get_c1(data.cell[n]);
              int c2 = mesh.get_c2(data.cell[n]);
              Scalar x1 = mesh.pos(0, c1, data.x1[n]);
              Scalar x2 = mesh.pos(1, c2, data.x2[n]);
              ptcoutput.x1[idx] = x1;
              ptcoutput.x2[idx] = x2;
              ptcoutput.x3[idx] = data.x3[n];
              ptcoutput.p1[idx] = data.p1[n];
              ptcoutput.p2[idx] = data.p2[n];
              ptcoutput.p3[idx] = data.p3[n];
              idx += 1;
            }
          }
        }
      }
      Logger::print_info("Writing {} tracked {}", idx,
                         NameStr((ParticleType)sp));
      DataSet data_x1 = datafile.createDataSet<float>(
          NameStr((ParticleType)sp) + "_x1", DataSpace{idx});
      data_x1.write(ptcoutput.x1);
      DataSet data_x2 = datafile.createDataSet<float>(
          NameStr((ParticleType)sp) + "_x2", DataSpace{idx});
      data_x2.write(ptcoutput.x2);
      DataSet data_x3 = datafile.createDataSet<float>(
          NameStr((ParticleType)sp) + "_x3", DataSpace{idx});
      data_x3.write(ptcoutput.x3);
      DataSet data_p1 = datafile.createDataSet<float>(
          NameStr((ParticleType)sp) + "_p1", DataSpace{idx});
      data_p1.write(ptcoutput.p1);
      DataSet data_p2 = datafile.createDataSet<float>(
          NameStr((ParticleType)sp) + "_p2", DataSpace{idx});
      data_p2.write(ptcoutput.p2);
      DataSet data_p3 = datafile.createDataSet<float>(
          NameStr((ParticleType)sp) + "_p3", DataSpace{idx});
      data_p3.write(ptcoutput.p3);
    }
  }
}

void
cu_data_exporter::write_output(cu_sim_data &data, uint32_t timestep,
                               double time) {
  if (!checkDirectories()) createDirectories();
  File datafile(fmt::format("{}{}{:06d}.h5", outputDirectory,
                            filePrefix, timestep)
                    .c_str(),
                File::ReadWrite | File::Create | File::Truncate);
  for (auto &f : m_fields_1d) {
  }
  for (auto &f : m_fields_2d) {
    int components = f.f.size();
    for (unsigned int n = 0; n < f.field.size(); n++) {
      if (components == 1) {
        auto fptr = dynamic_cast<cu_scalar_field<Scalar> *>(f.field[n]);
        // Logger::print_info("grid is {}x{}", fptr->grid().extent().x,
        //                    fptr->grid().extent().y);
        if (f.sync) fptr->sync_to_host();
        auto &mesh = fptr->grid().mesh();
#pragma omp simd collapse(2)
        for (int j = 0; j < mesh.reduced_dim(1);
             j += downsample_factor) {
          for (int i = 0; i < mesh.reduced_dim(0);
               i += downsample_factor) {
            f.f[0][j / downsample_factor + mesh.guard[1] +
                   m_submesh_out[n].offset[1]]
               [i / downsample_factor + mesh.guard[0] +
                m_submesh_out[n].offset[0]] =
                (*fptr)(i + mesh.guard[0], j + mesh.guard[1]);
          }
        }
      } else if (components == 3) {
        auto fptr = dynamic_cast<cu_vector_field<Scalar> *>(f.field[n]);
        if (f.sync) fptr->sync_to_host();
        auto &mesh = fptr->grid().mesh();
        // Logger::print_debug("offset[0] is {}, offset[1] is {}", m_submesh_out[n].offset[0],
        //                     m_submesh_out[n].offset[1]);
#pragma omp simd collapse(2)
        for (int j = 0; j < mesh.reduced_dim(1);
             j += downsample_factor) {
          for (int i = 0; i < mesh.reduced_dim(0);
               i += downsample_factor) {
            f.f[0][j / downsample_factor + mesh.guard[1] +
                   m_submesh_out[n].offset[1]]
               [i / downsample_factor + mesh.guard[0] +
                m_submesh_out[n].offset[0]] =
                (*fptr)(0, i + mesh.guard[0], j + mesh.guard[1]);
            f.f[1][j / downsample_factor + mesh.guard[1] +
                   m_submesh_out[n].offset[1]]
               [i / downsample_factor + mesh.guard[0] +
                m_submesh_out[n].offset[0]] =
                (*fptr)(1, i + mesh.guard[0], j + mesh.guard[1]);
            f.f[2][j / downsample_factor + mesh.guard[1] +
                   m_submesh_out[n].offset[1]]
               [i / downsample_factor + mesh.guard[0] +
                m_submesh_out[n].offset[0]] =
                (*fptr)(2, i + mesh.guard[0], j + mesh.guard[1]);
          }
        }
      }
    }
    if (components == 1) {
      // Logger::print_info("Creating dataset for {}", f.name);
      DataSet data = datafile.createDataSet<float>(
          f.name, DataSpace::From(f.f[0]));
      data.write(f.f[0]);
    } else {
      // Logger::print_info("Creating dataset for {}", f.name);
      for (int n = 0; n < components; n++) {
        // Logger::print_info("{}, {}", n, f.f[n].size());
        DataSet data = datafile.createDataSet<float>(
            f.name + std::to_string(n + 1), DataSpace::From(f.f[n]));
        // Logger::print_info("dataset created");
        data.write(f.f[n]);
      }
    }
  }
}

void
cu_data_exporter::prepare_output(cu_sim_data &data) {
  set_mesh(data);
  add_field("E", data.E, true);
  add_field("B", data.B, true);
  add_field("B_bg", data.Bbg, true);
  add_field("J", data.J, true);
  add_field("flux", data.flux, true);
  add_field("Rho_e", data.Rho[0], true);
  add_field("Rho_p", data.Rho[1], true);
  if (data.env.params().num_species > 2)
    add_field("Rho_i", data.Rho[2], true);
  add_field("photon_produced", data.photon_produced, true);
  add_field("pair_produced", data.pair_produced, true);
  add_field("photon_num", data.photon_num, true);
  add_field("divE", data.divE, true);
  add_field("divB", data.divB, true);
  add_field("EdotBavg", data.EdotB, true);
}

void
cu_data_exporter::writeXMFStep(std::ofstream &fs, uint32_t step,
                               double time) {
  std::string dim_str;
  auto &mesh = grid->mesh();
  if (grid->dim() == 3) {
    dim_str = fmt::format("{} {} {}", mesh.dims[2], mesh.dims[1],
                          mesh.dims[0]);
  } else if (grid->dim() == 2) {
    dim_str = fmt::format("{} {}", mesh.dims[1], mesh.dims[0]);
  } else if (grid->dim() == 1) {
    dim_str = fmt::format("{} 1", mesh.dims[0]);
  }

  fs << "<Grid Name=\"quadmesh\" Type=\"Uniform\">" << std::endl;
  fs << "  <Time Type=\"Single\" Value=\"" << time << "\"/>"
     << std::endl;
  if (grid->dim() == 3) {
    fs << "  <Topology Type=\"3DSMesh\" NumberOfElements=\"" << dim_str
       << "\"/>" << std::endl;
    fs << "  <Geometry GeometryType=\"X_Y_Z\">" << std::endl;
  } else if (grid->dim() == 2) {
    fs << "  <Topology Type=\"2DSMesh\" NumberOfElements=\"" << dim_str
       << "\"/>" << std::endl;
    fs << "  <Geometry GeometryType=\"X_Y\">" << std::endl;
  } else if (grid->dim() == 1) {
    fs << "  <Topology Type=\"2DSMesh\" NumberOfElements=\"" << dim_str
       << "\"/>" << std::endl;
    fs << "  <Geometry GeometryType=\"X_Y\">" << std::endl;
  }
  fs << "    <DataItem Dimensions=\"" << dim_str
     << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
     << std::endl;
  fs << "      mesh.h5:x1" << std::endl;
  fs << "    </DataItem>" << std::endl;
  if (grid->dim() >= 2) {
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << "      mesh.h5:x2" << std::endl;
    fs << "    </DataItem>" << std::endl;
  }
  if (grid->dim() >= 3) {
    fs << "    <DataItem Dimensions=\"" << dim_str
       << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
       << std::endl;
    fs << "      mesh.h5:x3" << std::endl;
    fs << "    </DataItem>" << std::endl;
  }

  fs << "  </Geometry>" << std::endl;

  for (auto &f : m_fields_2d) {
    if (f.f.size() == 1) {
      fs << "  <Attribute Name=\"" << f.name
         << "\" Center=\"Node\" AttributeType=\"Scalar\">" << std::endl;
      fs << "    <DataItem Dimensions=\"" << dim_str
         << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
         << std::endl;
      fs << fmt::format("      {}{:06d}.h5:{}", filePrefix, step,
                        f.name)
         << std::endl;
      fs << "    </DataItem>" << std::endl;
      fs << "  </Attribute>" << std::endl;
    } else if (f.f.size() == 3) {
      for (int i = 0; i < 3; i++) {
        fs << "  <Attribute Name=\"" << f.name << i + 1
           << "\" Center=\"Node\" AttributeType=\"Scalar\">"
           << std::endl;
        fs << "    <DataItem Dimensions=\"" << dim_str
           << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
           << std::endl;
        fs << fmt::format("      {}{:06d}.h5:{}{}", filePrefix, step,
                          f.name, i + 1)
           << std::endl;
        fs << "    </DataItem>" << std::endl;
        fs << "  </Attribute>" << std::endl;
      }
    }
  }

  for (auto &f : m_fields_1d) {
    if (f.f.size() == 1) {
      fs << "  <Attribute Name=\"" << f.name
         << "\" Center=\"Node\" AttributeType=\"Scalar\">" << std::endl;
      fs << "    <DataItem Dimensions=\"" << dim_str
         << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
         << std::endl;
      fs << fmt::format("      {}{:06d}.h5:{}", filePrefix, step,
                        f.name)
         << std::endl;
      fs << "    </DataItem>" << std::endl;
      fs << "  </Attribute>" << std::endl;
    } else if (f.f.size() == 3) {
      for (int i = 0; i < 1; i++) {
        fs << "  <Attribute Name=\"" << f.name << i + 1
           << "\" Center=\"Node\" AttributeType=\"Scalar\">"
           << std::endl;
        fs << "    <DataItem Dimensions=\"" << dim_str
           << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
           << std::endl;
        fs << fmt::format("      {}{:06d}.h5:{}{}", filePrefix, step,
                          f.name, i + 1)
           << std::endl;
        fs << "    </DataItem>" << std::endl;
        fs << "  </Attribute>" << std::endl;
      }
    }
  }
  // for (auto &sf : dbScalars2d) {
  //   fs << "  <Attribute Name=\"" << sf.name
  //      << "\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}", filePrefix, step,
  //   sf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  // }
  // for (auto &sf : dbScalars3d) {
  //   fs << "  <Attribute Name=\"" << sf.name
  //      << "\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}", filePrefix, step,
  //   sf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  // }
  // for (auto &vf : dbVectors2d) {
  //   fs << "  <Attribute Name=\"" << vf.name
  //      << "1\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}1", filePrefix, step,
  //                     vf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  //   fs << "  <Attribute Name=\"" << vf.name
  //      << "2\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}2", filePrefix, step,
  //                     vf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  //   fs << "  <Attribute Name=\"" << vf.name
  //      << "3\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}3", filePrefix, step,
  //                     vf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  // }

  // for (auto &vf : dbVectors3d) {
  //   fs << "  <Attribute Name=\"" << vf.name
  //      << "1\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}1", filePrefix, step,
  //                     vf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  //   fs << "  <Attribute Name=\"" << vf.name
  //      << "2\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}2", filePrefix, step,
  //                     vf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  //   fs << "  <Attribute Name=\"" << vf.name
  //      << "3\" Center=\"Node\" AttributeType=\"Scalar\">" <<
  //      std::endl;
  //   fs << "    <DataItem Dimensions=\"" << dim_str
  //      << "\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">"
  //      << std::endl;
  //   fs << fmt::format("      {}{:06d}.h5:{}3", filePrefix, step,
  //                     vf.name)
  //      << std::endl;
  //   fs << "    </DataItem>" << std::endl;
  //   fs << "  </Attribute>" << std::endl;
  // }

  fs << "</Grid>" << std::endl;
}

// Explicit instantiation of templates
template void cu_data_exporter::add_field<float>(
    const std::string &name, std::vector<cu_scalar_field<float>> &field,
    bool sync);
template void cu_data_exporter::add_field<float>(
    const std::string &name, std::vector<cu_vector_field<float>> &field,
    bool sync);
template void cu_data_exporter::add_field<double>(
    const std::string &name,
    std::vector<cu_scalar_field<double>> &field, bool sync);
template void cu_data_exporter::add_field<double>(
    const std::string &name,
    std::vector<cu_vector_field<double>> &field, bool sync);

template void cu_data_exporter::interpolate_field_values<float>(
    fieldoutput<2> &field, int components, const float &t);
template void cu_data_exporter::interpolate_field_values<double>(
    fieldoutput<2> &field, int components, const double &t);
template void hdf_exporter<cu_data_exporter>::add_array_output<float>(
    const std::string &name, multi_array<float> &array);

}  // namespace Aperture
