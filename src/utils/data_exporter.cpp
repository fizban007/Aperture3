#include "utils/data_exporter.h"
#include "utils/hdf_exporter_impl.hpp"
#include "sim_data.h"
#include "sim_environment.h"
#include "sim_params.h"
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
                                        int components, const T &t) {}

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
                    (downsample_factor * downsample_factor * downsample_factor);
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
                    (downsample_factor * downsample_factor * downsample_factor);

                field.f[2][k / downsample_factor + mesh.guard[2]]
                       [j / downsample_factor + mesh.guard[1]]
                       [i / downsample_factor + mesh.guard[0]] +=
                    (*fptr)(2, i + n1 + mesh.guard[0],
                            j + n2 + mesh.guard[1],
                            k + n3 + mesh.guard[2]) /
                    (downsample_factor * downsample_factor * downsample_factor);
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


}  // namespace Aperture
