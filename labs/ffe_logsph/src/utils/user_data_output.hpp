#ifndef __USER_DATA_OUTPUT_H_
#define __USER_DATA_OUTPUT_H_

#include "grids/grid_1dgr.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/data_exporter.h"
#include "utils/util_functions.h"
// #include <highfive/H5File.hpp>

// using namespace HighFive;

namespace Aperture {

void
user_write_field_output(sim_data& data, data_exporter& exporter,
                        uint32_t timestep, double time, hid_t file) {
  ADD_GRID_OUTPUT(
      exporter, data, "E1",
      {
        p(idx_out) =
            0.5 * (data.E(0, idx) + data.E(0, idx.x, idx.y + 1, 0));
      },
      file, timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "E2",
      {
        p(idx_out) =
            0.5 * (data.E(1, idx) + data.E(1, idx.x + 1, idx.y, 0));
      },
      file, timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "E3",
      {
        p(idx_out) =
            0.25 * (data.E(2, idx) + data.E(2, idx.x + 1, idx.y, 0) +
                    data.E(2, idx.x, idx.y + 1, 0) +
                    data.E(2, idx.x + 1, idx.y + 1, 0));
      },
      file, timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "B1",
      {
        p(idx_out) =
            0.5 * (data.B(0, idx) + data.B(0, idx.x - 1, idx.y, 0) +
                   data.Bbg(0, idx) + data.Bbg(0, idx.x - 1, idx.y, 0));
      },
      file, timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "B2",
      {
        p(idx_out) =
            0.5 * (data.B(1, idx) + data.B(1, idx.x, idx.y - 1, 0) +
                   data.Bbg(1, idx) + data.Bbg(1, idx.x, idx.y - 1, 0));
      },
      file, timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "B3",
      { p(idx_out) = data.B(2, idx) + data.Bbg(2, idx); }, file,
      timestep);
  // ADD_GRID_OUTPUT(
  //     exporter, data, "J1",
  //     {
  //       p(idx_out) =
  //           0.5 * (data.J(0, idx) + data.J(0, idx.x, idx.y + 1, 0));
  //     },
  //     file, timestep);
  // ADD_GRID_OUTPUT(
  //     exporter, data, "J2",
  //     {
  //       p(idx_out) =
  //           0.5 * (data.J(1, idx) + data.J(1, idx.x + 1, idx.y, 0));
  //     },
  //     file, timestep);
  // ADD_GRID_OUTPUT(
  //     exporter, data, "J3",
  //     {
  //       p(idx_out) =
  //           0.25 * (data.J(2, idx) + data.J(2, idx.x + 1, idx.y, 0) +
  //                   data.J(2, idx.x, idx.y + 1, 0) +
  //                   data.J(2, idx.x + 1, idx.y + 1, 0));
  //     },
  //     file, timestep);
}

}  // namespace Aperture

#endif  // __USER_DATA_OUTPUT_H_
