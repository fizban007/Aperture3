#include "utils/data_exporter.h"
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

}  // namespace Aperture
