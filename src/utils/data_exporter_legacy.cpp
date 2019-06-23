#include "utils/data_exporter.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "sim_params.h"
#include "utils/hdf_exporter_impl.hpp"
#include "utils/type_name.h"

#define H5_USE_BOOST

#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <time.h>

#include "visit_struct/visit_struct.hpp"

using namespace HighFive;

namespace Aperture {

template class hdf_exporter<data_exporter>;

data_exporter::data_exporter(SimParams &params, uint32_t &timestep)
    : hdf_exporter(params, timestep) {}

data_exporter::~data_exporter() {}

template <typename T>
void
data_exporter::add_field(const std::string &name,
                         scalar_field<T> &field) {
  add_field_output(name, TypeName<T>::Get(), 1, &field,
                   field.grid().dim(), false);
}

template <typename T>
void
data_exporter::add_field(const std::string &name,
                         vector_field<T> &field) {
  add_field_output(name, TypeName<T>::Get(), VECTOR_DIM, &field,
                   field.grid().dim(), false);
}

template <typename T>
void
data_exporter::interpolate_field_values(fieldoutput<1> &field,
                                        int components, const T &t) {
  if (components == 1) {
    auto fptr = dynamic_cast<scalar_field<T> *>(field.field);
    auto &mesh = fptr->grid().mesh();
    for (int i = 0; i < mesh.reduced_dim(0); i += downsample_factor) {
      field.f[0][i / downsample_factor + mesh.guard[0]] = 0.0;
      for (int n1 = 0; n1 < downsample_factor; n1++) {
        field.f[0][i / downsample_factor + mesh.guard[0]] +=
            (*fptr)(i + n1 + mesh.guard[0]) / downsample_factor;
      }
    }
  } else if (components == 3) {
    auto fptr = dynamic_cast<vector_field<T> *>(field.field);
    auto &mesh = fptr->grid().mesh();
    for (int i = 0; i < mesh.reduced_dim(0); i += downsample_factor) {
      field.f[0][i / downsample_factor + mesh.guard[0]] = 0.0;
      // field.f[1][i / downsample_factor + mesh.guard[0]] = 0.0;
      // field.f[2][i / downsample_factor + mesh.guard[0]] = 0.0;
      for (int n1 = 0; n1 < downsample_factor; n1++) {
        field.f[0][i / downsample_factor + mesh.guard[0]] +=
            (*fptr)(0, i + n1 + mesh.guard[0]) / downsample_factor;
      }
    }
  }
}

template <typename T>
void
data_exporter::interpolate_field_values(fieldoutput<2> &field,
                                        int components, const T &t) {
  if (components == 1) {
    auto fptr = dynamic_cast<scalar_field<T> *>(field.field);
    auto &mesh = fptr->grid().mesh();
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
    auto fptr = dynamic_cast<vector_field<T> *>(field.field);
    auto &mesh = fptr->grid().mesh();
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
data_exporter::interpolate_field_values(fieldoutput<3> &field,
                                        int components, const T &t) {
  if (components == 1) {
    auto fptr = dynamic_cast<scalar_field<T> *>(field.field);
    auto &mesh = fptr->grid().mesh();
    for (int k = 0; k < mesh.reduced_dim(2); k += downsample_factor) {
      for (int j = 0; j < mesh.reduced_dim(1); j += downsample_factor) {
        for (int i = 0; i < mesh.reduced_dim(0);
             i += downsample_factor) {
          field.f[0][k / downsample_factor + mesh.guard[2]]
                 [j / downsample_factor + mesh.guard[1]]
                 [i / downsample_factor + mesh.guard[0]] = 0.0;

          for (int n3 = 0; n3 < downsample_factor; n3++) {
            for (int n2 = 0; n2 < downsample_factor; n2++) {
              for (int n1 = 0; n1 < downsample_factor; n1++) {
                field.f[0][k / downsample_factor + mesh.guard[2]]
                       [j / downsample_factor + mesh.guard[1]]
                       [i / downsample_factor + mesh.guard[0]] +=
                    (*fptr)(i + n1 + mesh.guard[0],
                            j + n2 + mesh.guard[1],
                            k + n3 + mesh.guard[2]) /
                    (downsample_factor * downsample_factor *
                     downsample_factor);
              }
            }
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
    auto fptr = dynamic_cast<vector_field<T> *>(field.field);
    if (field.sync) fptr->sync_to_host();
    auto &mesh = fptr->grid().mesh();
    for (int k = 0; k < mesh.reduced_dim(2); k += downsample_factor) {
      for (int j = 0; j < mesh.reduced_dim(1); j += downsample_factor) {
        for (int i = 0; i < mesh.reduced_dim(0);
             i += downsample_factor) {
          field.f[0][k / downsample_factor + mesh.guard[2]]
                 [j / downsample_factor + mesh.guard[1]]
                 [i / downsample_factor + mesh.guard[0]] = 0.0;
          field.f[1][k / downsample_factor + mesh.guard[2]]
                 [j / downsample_factor + mesh.guard[1]]
                 [i / downsample_factor + mesh.guard[0]] = 0.0;
          field.f[2][k / downsample_factor + mesh.guard[2]]
                 [j / downsample_factor + mesh.guard[1]]
                 [i / downsample_factor + mesh.guard[0]] = 0.0;
          for (int n3 = 0; n3 < downsample_factor; n3++) {
            for (int n2 = 0; n2 < downsample_factor; n2++) {
              for (int n1 = 0; n1 < downsample_factor; n1++) {
                field.f[0][k / downsample_factor + mesh.guard[2]]
                       [j / downsample_factor + mesh.guard[1]]
                       [i / downsample_factor + mesh.guard[0]] +=
                    (*fptr)(0, i + n1 + mesh.guard[0],
                            j + n2 + mesh.guard[1],
                            k + n3 + mesh.guard[2]) /
                    (downsample_factor * downsample_factor *
                     downsample_factor);
                // std::cout << vf.f1[j / downsample_factor +
                // mesh.guard[1]
                //     [i / downsample_factor + mesh.guard[0]] <<
                // std::endl;
                field.f[1][k / downsample_factor + mesh.guard[2]]
                       [j / downsample_factor + mesh.guard[1]]
                       [i / downsample_factor + mesh.guard[0]] +=
                    (*fptr)(1, i + n1 + mesh.guard[0],
                            j + n2 + mesh.guard[1],
                            k + n3 + mesh.guard[2]) /
                    (downsample_factor * downsample_factor *
                     downsample_factor);

                field.f[2][k / downsample_factor + mesh.guard[2]]
                       [j / downsample_factor + mesh.guard[1]]
                       [i / downsample_factor + mesh.guard[0]] +=
                    (*fptr)(2, i + n1 + mesh.guard[0],
                            j + n2 + mesh.guard[1],
                            k + n3 + mesh.guard[2]) /
                    (downsample_factor * downsample_factor *
                     downsample_factor);
              }
            }
          }
        }
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

void
data_exporter::write_particles(uint32_t step, double time) {
  auto &mesh = grid->mesh();
  for (auto &ptcoutput : dbPtcData1d) {
    particles_t *ptc = dynamic_cast<particles_t *>(ptcoutput.ptc);
    if (ptc != nullptr) {
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
          if (!ptc->is_empty(n) &&
              ptc->check_flag(n, ParticleFlag::tracked) &&
              ptc->check_type(n) == (ParticleType)sp &&
              idx < MAX_TRACKED) {
            Scalar x =
                mesh.pos(0, ptc->data().cell[n], ptc->data().x1[n]);
            ptcoutput.x[idx] = x;
            ptcoutput.p[idx] = ptc->data().p1[n];
            idx += 1;
          }
        }
        DataSet data_x =
            datafile.createDataSet<float>(name_x, DataSpace{idx});
        data_x.write(ptcoutput.x);
        DataSet data_p =
            datafile.createDataSet<float>(name_p, DataSpace{idx});
        data_p.write(ptcoutput.p);

        // Logger::print_info("Written {} tracked particles", idx);
      }
      // hsize_t sizes[1] = {idx};
      // H5::DataSpace space(1, sizes);
      // H5::DataSet *dataset_x = new H5::DataSet(file->createDataSet(
      //     name_x, H5::PredType::NATIVE_FLOAT, space));
      // dataset_x->write((void *)ds.data_x.data(),
      //                  H5::PredType::NATIVE_FLOAT);
      // H5::DataSet *dataset_p = new H5::DataSet(file->createDataSet(
      //     name_p, H5::PredType::NATIVE_FLOAT, space));
      // dataset_p->write((void *)ds.data_p.data(),
      //                  H5::PredType::NATIVE_FLOAT);

      // delete dataset_x;
      // delete dataset_p;
    }
  }
}

// Explicit instantiation
template void data_exporter::add_field<float>(
    const std::string &name, scalar_field<float> &field);
template void data_exporter::add_field<float>(
    const std::string &name, vector_field<float> &field);
template void data_exporter::add_field<double>(
    const std::string &name, scalar_field<double> &field);
template void data_exporter::add_field<double>(
    const std::string &name, vector_field<double> &field);
template void hdf_exporter<data_exporter>::add_array_output<float>(
    const std::string &name, multi_array<float> &array);

}  // namespace Aperture
