#include "cu_data_exporter.h"
#include "core/constant_defs.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/core/cu_sim_data1d.h"
#include "cuda/core/cu_sim_environment.h"
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
    : hdf_exporter(params, timestep),
      m_ptc_p(params.max_ptc_number),
      m_ph_p(params.max_photon_number),
      m_ptc_x1(params.max_ptc_number),
      m_ph_x1(params.max_photon_number),
      m_ptc_cell(params.max_ptc_number),
      m_ph_cell(params.max_photon_number),
      m_ptc_flag(params.max_ptc_number),
      m_ph_flag(params.max_photon_number) {}

cu_data_exporter::~cu_data_exporter() {}

template <typename T>
void
cu_data_exporter::add_field(const std::string &name,
                            cu_scalar_field<T> &field, bool sync) {
  add_field_output(name, TypeName<T>::Get(), 1, &field,
                   field.grid().dim(), sync);
}

template <typename T>
void
cu_data_exporter::add_field(const std::string &name,
                            cu_vector_field<T> &field, bool sync) {
  add_field_output(name, TypeName<T>::Get(), VECTOR_DIM, &field,
                   field.grid().dim(), sync);
}

void
cu_data_exporter::write_snapshot(cu_sim_environment &env,
                                 cu_sim_data &data, uint32_t timestep) {
  File snapshotfile(
      // fmt::format("{}snapshot{:06d}.h5", outputDirectory, timestep)
      fmt::format("{}snapshot.h5", outputDirectory).c_str(),
      File::ReadWrite | File::Create | File::Truncate);

  // Write background fields from environment
  // size_t grid_size = data.E.grid().size();
  size_t grid_size = data.E.data(0).size();
  data.Ebg.copy_to_host();
  DataSet data_bg_E1 =
      snapshotfile.createDataSet<Scalar>("bg_E1", DataSpace(grid_size));
  data_bg_E1.write((char *)data.Ebg.data(0).data());
  DataSet data_bg_E2 =
      snapshotfile.createDataSet<Scalar>("bg_E2", DataSpace(grid_size));
  data_bg_E2.write((char *)data.Ebg.data(1).data());
  DataSet data_bg_E3 =
      snapshotfile.createDataSet<Scalar>("bg_E3", DataSpace(grid_size));
  data_bg_E3.write((char *)data.Ebg.data(2).data());
  data.Bbg.copy_to_host();
  DataSet data_bg_B1 =
      snapshotfile.createDataSet<Scalar>("bg_B1", DataSpace(grid_size));
  data_bg_B1.write((char *)data.Bbg.data(0).data());
  DataSet data_bg_B2 =
      snapshotfile.createDataSet<Scalar>("bg_B2", DataSpace(grid_size));
  data_bg_B2.write((char *)data.Bbg.data(1).data());
  DataSet data_bg_B3 =
      snapshotfile.createDataSet<Scalar>("bg_B3", DataSpace(grid_size));
  data_bg_B3.write((char *)data.Bbg.data(2).data());

  // Write sim data
  // Write field values
  data.E.copy_to_host();
  DataSet data_E1 =
      snapshotfile.createDataSet<Scalar>("E1", DataSpace(grid_size));
  data_E1.write((char *)data.E.data(0).data());
  DataSet data_E2 =
      snapshotfile.createDataSet<Scalar>("E2", DataSpace(grid_size));
  data_E2.write((char *)data.E.data(1).data());
  DataSet data_E3 =
      snapshotfile.createDataSet<Scalar>("E3", DataSpace(grid_size));
  data_E3.write((char *)data.E.data(2).data());
  data.B.copy_to_host();
  DataSet data_B1 =
      snapshotfile.createDataSet<Scalar>("B1", DataSpace(grid_size));
  data_B1.write((char *)data.B.data(0).data());
  DataSet data_B2 =
      snapshotfile.createDataSet<Scalar>("B2", DataSpace(grid_size));
  data_B2.write((char *)data.B.data(1).data());
  DataSet data_B3 =
      snapshotfile.createDataSet<Scalar>("B3", DataSpace(grid_size));
  data_B3.write((char *)data.B.data(2).data());
  data.J.copy_to_host();
  DataSet data_J1 =
      snapshotfile.createDataSet<Scalar>("J1", DataSpace(grid_size));
  data_J1.write((char *)data.J.data(0).data());
  DataSet data_J2 =
      snapshotfile.createDataSet<Scalar>("J2", DataSpace(grid_size));
  data_J2.write((char *)data.J.data(1).data());
  DataSet data_J3 =
      snapshotfile.createDataSet<Scalar>("J3", DataSpace(grid_size));
  data_J3.write((char *)data.J.data(2).data());

  for (int i = 0; i < data.num_species; i++) {
    data.Rho[i].copy_to_host();
    DataSet data_Rho = snapshotfile.createDataSet<Scalar>(
        fmt::format("Rho{}", i), DataSpace(grid_size));
    data_Rho.write((char *)data.Rho[i].data().data());
  }
  DataSet data_devId = snapshotfile.createDataSet<int>(
      "devId", DataSpace::From(data.devId));
  data_devId.write(data.devId);

  // Write particle data
  size_t ptcNum = data.particles.number();
  DataSet data_ptcNum = snapshotfile.createDataSet<size_t>(
      "ptcNum", DataSpace::From(ptcNum));
  data_ptcNum.write(ptcNum);
  Logger::print_info("Writing {} particles to snapshot", ptcNum);

  size_t phNum = data.photons.number();
  DataSet data_phNum = snapshotfile.createDataSet<size_t>(
      "phNum", DataSpace::From(phNum));
  data_phNum.write(phNum);
  Logger::print_info("Writing {} photons to snapshot", phNum);

  std::vector<double> buffer(std::max(ptcNum, phNum));
  visit_struct::for_each(
      data.particles.data(),
      [&snapshotfile, &buffer, &ptcNum](const char *name, auto &x) {
        typedef
            typename std::remove_reference<decltype(*x)>::type x_type;
        DataSet ptc_data = snapshotfile.createDataSet<x_type>(
            fmt::format("ptc_{}", name), DataSpace(ptcNum));
        cudaMemcpy(buffer.data(), x, ptcNum * sizeof(x_type),
                   cudaMemcpyDeviceToHost);
        ptc_data.write(reinterpret_cast<x_type *>(buffer.data()));
      });
  visit_struct::for_each(
      data.photons.data(),
      [&snapshotfile, &buffer, &phNum](const char *name, auto &x) {
        typedef
            typename std::remove_reference<decltype(*x)>::type x_type;
        DataSet ph_data = snapshotfile.createDataSet<x_type>(
            fmt::format("ph_{}", name), DataSpace(phNum));
        cudaMemcpy(buffer.data(), x, phNum * sizeof(x_type),
                   cudaMemcpyDeviceToHost);
        ph_data.write(reinterpret_cast<x_type *>(buffer.data()));
      });

  // Write current simulation timestep and other info
  DataSet data_timestep = snapshotfile.createDataSet<uint32_t>(
      "timestep", DataSpace::From(timestep));
  data_timestep.write(timestep);
}

void
cu_data_exporter::load_from_snapshot(cu_sim_environment &env,
                                     cu_sim_data &data,
                                     uint32_t &timestep) {
  File snapshotfile(
      // fmt::format("{}snapshot{:06d}.h5", outputDirectory, timestep)
      fmt::format("{}snapshot.h5", outputDirectory).c_str(),
      File::ReadOnly);

  // size_t grid_size = data.E.grid().size();
  size_t ptcNum, phNum;
  int devId;

  // Read the scalars first
  DataSet data_timestep = snapshotfile.getDataSet("timestep");
  data_timestep.read(timestep);
  DataSet data_ptcNum = snapshotfile.getDataSet("ptcNum");
  data_ptcNum.read(ptcNum);
  DataSet data_phNum = snapshotfile.getDataSet("phNum");
  data_phNum.read(phNum);
  DataSet data_devId = snapshotfile.getDataSet("devId");
  data_devId.read(devId);

  // Read particle data
  std::vector<double> buffer(std::max(ptcNum, phNum));
  data.particles.set_num(ptcNum);
  data.photons.set_num(phNum);

  visit_struct::for_each(
      data.particles.data(),
      [&snapshotfile, &buffer, &ptcNum](const char *name, auto &x) {
        typedef
            typename std::remove_reference<decltype(*x)>::type x_type;
        DataSet ptc_data =
            snapshotfile.getDataSet(fmt::format("ptc_{}", name));
        ptc_data.read(reinterpret_cast<x_type *>(buffer.data()));
        cudaMemcpy(x, buffer.data(), ptcNum * sizeof(x_type),
                   cudaMemcpyHostToDevice);
      });
  visit_struct::for_each(
      data.photons.data(),
      [&snapshotfile, &buffer, &phNum](const char *name, auto &x) {
        typedef
            typename std::remove_reference<decltype(*x)>::type x_type;
        DataSet ph_data =
            snapshotfile.getDataSet(fmt::format("ph_{}", name));
        ph_data.read(reinterpret_cast<x_type *>(buffer.data()));
        cudaMemcpy(x, buffer.data(), phNum * sizeof(x_type),
                   cudaMemcpyHostToDevice);
      });

  // Read field data
  DataSet data_bg_B1 = snapshotfile.getDataSet("bg_B1");
  data_bg_B1.read((char *)data.Bbg.data(0).data());
  DataSet data_bg_B2 = snapshotfile.getDataSet("bg_B2");
  data_bg_B2.read((char *)data.Bbg.data(1).data());
  DataSet data_bg_B3 = snapshotfile.getDataSet("bg_B3");
  data_bg_B3.read((char *)data.Bbg.data(2).data());
  DataSet data_bg_E1 = snapshotfile.getDataSet("bg_E1");
  data_bg_E1.read((char *)data.Ebg.data(0).data());
  DataSet data_bg_E2 = snapshotfile.getDataSet("bg_E2");
  data_bg_E2.read((char *)data.Ebg.data(1).data());
  DataSet data_bg_E3 = snapshotfile.getDataSet("bg_E3");
  data_bg_E3.read((char *)data.Ebg.data(2).data());

  data.Bbg.copy_to_device();
  data.Ebg.copy_to_device();

  DataSet data_B1 = snapshotfile.getDataSet("B1");
  data_B1.read((char *)data.B.data(0).data());
  DataSet data_B2 = snapshotfile.getDataSet("B2");
  data_B2.read((char *)data.B.data(1).data());
  DataSet data_B3 = snapshotfile.getDataSet("B3");
  data_B3.read((char *)data.B.data(2).data());
  DataSet data_E1 = snapshotfile.getDataSet("E1");
  data_E1.read((char *)data.E.data(0).data());
  DataSet data_E2 = snapshotfile.getDataSet("E2");
  data_E2.read((char *)data.E.data(1).data());
  DataSet data_E3 = snapshotfile.getDataSet("E3");
  data_E3.read((char *)data.E.data(2).data());
  DataSet data_J1 = snapshotfile.getDataSet("J1");
  data_J1.read((char *)data.J.data(0).data());
  DataSet data_J2 = snapshotfile.getDataSet("J2");
  data_J2.read((char *)data.J.data(1).data());
  DataSet data_J3 = snapshotfile.getDataSet("J3");
  data_J3.read((char *)data.J.data(2).data());
  data.B.copy_to_device();
  data.E.copy_to_device();
  data.J.copy_to_device();

  for (int i = 0; i < data.num_species; i++) {
    DataSet data_rho = snapshotfile.getDataSet(fmt::format("Rho{}", i));
    data_rho.read((char *)data.Rho[i].data().data());
    data.Rho[i].copy_to_device();
  }
}

template <typename T>
void
cu_data_exporter::interpolate_field_values(fieldoutput<1> &field,
                                           int components, const T &t) {
  if (components == 1) {
    auto fptr = dynamic_cast<cu_scalar_field<T> *>(field.field);
    if (field.sync) fptr->copy_to_host();
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
    if (field.sync) fptr->copy_to_host();
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
    if (field.sync) fptr->copy_to_host();
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
    if (field.sync) fptr->copy_to_host();
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
cu_data_exporter::write_particles(uint32_t step, double time) {
  auto &mesh = grid->mesh();
  for (auto &ptcoutput : dbPtcData1d) {
    if (ptcoutput.type == "ptc") {
      Particles_1D *ptc = dynamic_cast<Particles_1D *>(ptcoutput.ptc);
      if (ptc != nullptr) {
        Logger::print_info("Copying {} ptc from dev to host",
                           ptc->number());
        cudaMemcpy(m_ptc_x1.data(), ptc->data().x1,
                   sizeof(Pos_t) * ptc->number(),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(m_ptc_p.data(), ptc->data().p1,
                   sizeof(Scalar) * ptc->number(),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(m_ptc_cell.data(), ptc->data().cell,
                   sizeof(uint32_t) * ptc->number(),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(m_ptc_flag.data(), ptc->data().flag,
                   sizeof(uint32_t) * ptc->number(),
                   cudaMemcpyDeviceToHost);
        Logger::print_info("Writing tracked particles");
        File datafile(fmt::format("{}{}{:06d}.h5", outputDirectory,
                                  filePrefix, step)
                          .c_str(),
                      File::ReadWrite);
        for (int sp = 0; sp < m_params.num_species; sp++) {
          unsigned int idx = 0;
          std::string name_x = NameStr((ParticleType)sp) + "_x";
          std::string name_p = NameStr((ParticleType)sp) + "_p";
          for (Index_t n = 0; n < ptc->number(); n++) {
            if (m_ptc_cell[n] != MAX_CELL &&
                // ptc->check_flag(n, ParticleFlag::tracked) &&
                check_bit(m_ptc_flag[n], ParticleFlag::tracked) &&
                // ptc->check_type(n) == (ParticleType)sp &&
                get_ptc_type(m_ptc_flag[n]) == sp &&
                idx < MAX_TRACKED) {
              Scalar x = mesh.pos(0, m_ptc_cell[n], m_ptc_x1[n]);
              ptcoutput.x[idx] = x;
              ptcoutput.p[idx] = m_ptc_p[n];
              idx += 1;
            }
          }
          DataSet data_x =
              datafile.createDataSet<float>(name_x, DataSpace{idx});
          data_x.write(ptcoutput.x);
          DataSet data_p =
              datafile.createDataSet<float>(name_p, DataSpace{idx});
          data_p.write(ptcoutput.p);

          Logger::print_info("Written {} tracked particles", idx);
        }
      }
    } else if (ptcoutput.type == "photon") {
      Photons_1D *ptc = dynamic_cast<Photons_1D *>(ptcoutput.ptc);
      if (ptc != nullptr) {
        Logger::print_info("Copying {} photons from dev to host",
                           ptc->number());
        cudaMemcpy(m_ph_x1.data(), ptc->data().x1,
                   sizeof(Pos_t) * ptc->number(),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(m_ph_p.data(), ptc->data().p1,
                   sizeof(Scalar) * ptc->number(),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(m_ph_cell.data(), ptc->data().cell,
                   sizeof(uint32_t) * ptc->number(),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(m_ph_flag.data(), ptc->data().flag,
                   sizeof(uint32_t) * ptc->number(),
                   cudaMemcpyDeviceToHost);
        Logger::print_info("Writing tracked particles");
        File datafile(fmt::format("{}{}{:06d}.h5", outputDirectory,
                                  filePrefix, step)
                          .c_str(),
                      File::ReadWrite);
        unsigned int idx = 0;
        std::string name_x = "photons_x";
        std::string name_p = "photons_p";
        for (Index_t n = 0; n < ptc->number(); n++) {
          if (m_ph_cell[n] != MAX_CELL &&
              // ptc->check_flag(n, ParticleFlag::tracked) &&
              check_bit(m_ph_flag[n], PhotonFlag::tracked) &&
              idx < MAX_TRACKED) {
            Scalar x = mesh.pos(0, m_ph_cell[n], m_ph_x1[n]);
            ptcoutput.x[idx] = x;
            ptcoutput.p[idx] = m_ph_p[n];
            idx += 1;
          }
        }
        DataSet data_x =
            datafile.createDataSet<float>(name_x, DataSpace{idx});
        data_x.write(ptcoutput.x);
        DataSet data_p =
            datafile.createDataSet<float>(name_p, DataSpace{idx});
        data_p.write(ptcoutput.p);

        Logger::print_info("Written {} tracked photons", idx);
      }
    }
  }
}

// Explicit instantiation of templates
template void cu_data_exporter::add_field<float>(
    const std::string &name, cu_scalar_field<float> &field, bool sync);
template void cu_data_exporter::add_field<float>(
    const std::string &name, cu_vector_field<float> &field, bool sync);
template void cu_data_exporter::add_field<double>(
    const std::string &name, cu_scalar_field<double> &field, bool sync);
template void cu_data_exporter::add_field<double>(
    const std::string &name, cu_vector_field<double> &field, bool sync);

template void cu_data_exporter::interpolate_field_values<float>(
    fieldoutput<2> &field, int components, const float &t);
template void cu_data_exporter::interpolate_field_values<double>(
    fieldoutput<2> &field, int components, const double &t);
template void hdf_exporter<cu_data_exporter>::add_array_output<float>(
    const std::string &name, multi_array<float> &array);

}  // namespace Aperture

// namespace Aperture
