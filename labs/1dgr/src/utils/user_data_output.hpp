#ifndef __USER_DATA_OUTPUT_H_
#define __USER_DATA_OUTPUT_H_

#include "sim_data.h"
#include "sim_environment.h"
#include "utils/data_exporter.h"
#include <highfive/H5File.hpp>

using namespace HighFive;

namespace Aperture {

void
user_write_field_output(sim_data& data, data_exporter& exporter,
                        uint32_t timestep, double time, File& file) {
  ADD_GRID_OUTPUT(
      exporter, data, "E1", { p(idx_out) = data.E(0, idx); }, file);
  ADD_GRID_OUTPUT(
      exporter, data, "J1", { p(idx_out) = data.J(0, idx); }, file);
  ADD_GRID_OUTPUT(
      exporter, data, "Rho_e", { p(idx_out) = data.Rho[0](0, idx); }, file);
  ADD_GRID_OUTPUT(
      exporter, data, "Rho_p", { p(idx_out) = data.Rho[1](0, idx); }, file);
}

void
user_write_ptc_output(sim_data& data, data_exporter& exporter,
                      uint32_t timestep, double time, File& file) {
  exporter.add_ptc_uint_output(
      data, "ptc_id", data.particles.tracked_number(),
      [](sim_data& data, std::vector<uint32_t>& v, uint32_t n) {
        v[n] = data.particles.tracked_data().id[n];
      },
      file);
  exporter.add_ptc_uint_output(
      data, "ptc_flag", data.particles.tracked_number(),
      [](sim_data& data, std::vector<uint32_t>& v, uint32_t n) {
        v[n] = data.particles.tracked_data().flag[n];
      },
      file);
  exporter.add_ptc_float_output(
      data, "ptc_p1", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n) {
        v[n] = data.particles.tracked_data().p1[n];
      },
      file);
  exporter.add_ptc_float_output(
      data, "ptc_u0", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n) {
        v[n] = data.particles.tracked_data().p2[n];
      },
      file);
  auto& mesh = data.env.local_grid().mesh();
  exporter.add_ptc_float_output(
      data, "ptc_x1", data.particles.tracked_number(),
      [&mesh](sim_data& data, std::vector<float>& v, uint32_t n) {
        auto cell = data.particles.tracked_data().cell[n];
        auto x = data.particles.tracked_data().x1[n];
        v[n] = mesh.pos(0, mesh.get_c1(cell), x);
      },
      file);
  exporter.add_ptc_uint_output(
      data, "ph_id", data.photons.tracked_number(),
      [](sim_data& data, std::vector<uint32_t>& v, uint32_t n) {
        v[n] = data.photons.tracked_data().id[n];
      },
      file);

  exporter.add_ptc_float_output(
      data, "ph_p1", data.photons.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n) {
        v[n] = data.photons.tracked_data().p1[n];
      },
      file);
  exporter.add_ptc_float_output(
      data, "ph_u0", data.photons.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n) {
        v[n] = data.photons.tracked_data().p2[n];
      },
      file);
  exporter.add_ptc_float_output(
      data, "ph_p3", data.photons.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n) {
        v[n] = data.photons.tracked_data().p3[n];
      },
      file);
  exporter.add_ptc_float_output(
      data, "ph_x1", data.photons.tracked_number(),
      [&mesh](sim_data& data, std::vector<float>& v, uint32_t n) {
        auto cell = data.photons.tracked_data().cell[n];
        auto x = data.photons.tracked_data().x1[n];
        v[n] = mesh.pos(0, mesh.get_c1(cell), x);
      },
      file);
}

}  // namespace Aperture

#endif  // __USER_DATA_OUTPUT_H_
