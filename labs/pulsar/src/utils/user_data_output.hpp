#ifndef __USER_DATA_OUTPUT_H_
#define __USER_DATA_OUTPUT_H_

#include "grids/grid_1dgr.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/data_exporter.h"
#include "utils/util_functions.h"
#include <highfive/H5File.hpp>

using namespace HighFive;

namespace Aperture {

void
user_write_field_output(sim_data& data, data_exporter& exporter,
                        uint32_t timestep, double time, File& file) {
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
  ADD_GRID_OUTPUT(
      exporter, data, "J1",
      {
        p(idx_out) =
            0.5 * (data.J(0, idx) + data.J(0, idx.x, idx.y + 1, 0));
      },
      file, timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "J2",
      {
        p(idx_out) =
            0.5 * (data.J(1, idx) + data.J(1, idx.x + 1, idx.y, 0));
      },
      file, timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "J3",
      {
        p(idx_out) =
            0.25 * (data.J(2, idx) + data.J(2, idx.x + 1, idx.y, 0) +
                    data.J(2, idx.x, idx.y + 1, 0) +
                    data.J(2, idx.x + 1, idx.y + 1, 0));
      },
      file, timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "Rho_e", { p(idx_out) = data.Rho[0](0, idx); },
      file, timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "Rho_p", { p(idx_out) = data.Rho[1](0, idx); },
      file, timestep);
  if (data.env.params().num_species > 2) {
    ADD_GRID_OUTPUT(
        exporter, data, "Rho_i", { p(idx_out) = data.Rho[2](0, idx); },
        file, timestep);
  }
  ADD_GRID_OUTPUT(
      exporter, data, "photon_produced",
      { p(idx_out) = data.photon_produced(idx); }, file, timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "pair_produced",
      { p(idx_out) = data.pair_produced(idx); }, file, timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "photon_num",
      { p(idx_out) = data.photon_num(idx); }, file, timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "divE", { p(idx_out) = data.divE(idx); }, file,
      timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "divB", { p(idx_out) = data.divB(idx); }, file,
      timestep);
  ADD_GRID_OUTPUT(
      exporter, data, "EdotB_avg", { p(idx_out) = data.EdotB(idx); },
      file, timestep);
}

void
user_write_ptc_output(sim_data& data, data_exporter& exporter,
                      uint32_t timestep, double time, File& file) {
  exporter.add_ptc_uint_output(
      data, "electron_id", data.particles.tracked_number(),
      [](sim_data& data, std::vector<uint32_t>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::electron) {
          v[nsb] = data.particles.tracked_data().id[n];
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "electron_p1", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::electron) {
          v[nsb] = data.particles.tracked_data().p1[n];
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "electron_p2", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::electron) {
          v[nsb] = data.particles.tracked_data().p2[n];
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "electron_p3", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::electron) {
          v[nsb] = data.particles.tracked_data().p3[n];
          nsb += 1;
        }
      },
      file, timestep);
  auto& mesh = data.env.local_grid().mesh();
  exporter.add_ptc_float_output(
      data, "electron_x1", data.particles.tracked_number(),
      [&mesh](sim_data& data, std::vector<float>& v, uint32_t n,
              uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::electron) {
          auto cell = data.particles.tracked_data().cell[n];
          auto x = data.particles.tracked_data().x1[n];
          v[nsb] = mesh.pos(0, mesh.get_c1(cell), x);
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "electron_x2", data.particles.tracked_number(),
      [&mesh](sim_data& data, std::vector<float>& v, uint32_t n,
              uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::electron) {
          auto cell = data.particles.tracked_data().cell[n];
          auto x = data.particles.tracked_data().x2[n];
          v[nsb] = mesh.pos(0, mesh.get_c2(cell), x);
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "electron_x3", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::electron) {
          auto x = data.particles.tracked_data().x3[n];
          v[nsb] = x;
          nsb += 1;
        }
      },
      file, timestep);

  exporter.add_ptc_uint_output(
      data, "positron_id", data.particles.tracked_number(),
      [](sim_data& data, std::vector<uint32_t>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::positron) {
          v[nsb] = data.particles.tracked_data().id[n];
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "positron_p1", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::positron) {
          v[nsb] = data.particles.tracked_data().p1[n];
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "positron_p2", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::positron) {
          v[nsb] = data.particles.tracked_data().p2[n];
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "positron_p3", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::positron) {
          v[nsb] = data.particles.tracked_data().p3[n];
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "positron_x1", data.particles.tracked_number(),
      [&mesh](sim_data& data, std::vector<float>& v, uint32_t n,
              uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::positron) {
          auto cell = data.particles.tracked_data().cell[n];
          auto x = data.particles.tracked_data().x1[n];
          v[nsb] = mesh.pos(0, mesh.get_c1(cell), x);
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "positron_x2", data.particles.tracked_number(),
      [&mesh](sim_data& data, std::vector<float>& v, uint32_t n,
              uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::positron) {
          auto cell = data.particles.tracked_data().cell[n];
          auto x = data.particles.tracked_data().x2[n];
          v[nsb] = mesh.pos(0, mesh.get_c2(cell), x);
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "positron_x3", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::positron) {
          auto x = data.particles.tracked_data().x3[n];
          v[nsb] = x;
          nsb += 1;
        }
      },
      file, timestep);

  exporter.add_ptc_uint_output(
      data, "ion_id", data.particles.tracked_number(),
      [](sim_data& data, std::vector<uint32_t>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::ion) {
          v[nsb] = data.particles.tracked_data().id[n];
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "ion_p1", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::ion) {
          v[nsb] = data.particles.tracked_data().p1[n];
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "ion_p2", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::ion) {
          v[nsb] = data.particles.tracked_data().p2[n];
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "ion_p3", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::ion) {
          v[nsb] = data.particles.tracked_data().p3[n];
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "ion_x1", data.particles.tracked_number(),
      [&mesh](sim_data& data, std::vector<float>& v, uint32_t n,
              uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::ion) {
          auto cell = data.particles.tracked_data().cell[n];
          auto x = data.particles.tracked_data().x1[n];
          v[nsb] = mesh.pos(0, mesh.get_c1(cell), x);
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "ion_x2", data.particles.tracked_number(),
      [&mesh](sim_data& data, std::vector<float>& v, uint32_t n,
              uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::ion) {
          auto cell = data.particles.tracked_data().cell[n];
          auto x = data.particles.tracked_data().x2[n];
          v[nsb] = mesh.pos(0, mesh.get_c2(cell), x);
          nsb += 1;
        }
      },
      file, timestep);
  exporter.add_ptc_float_output(
      data, "ion_x3", data.particles.tracked_number(),
      [](sim_data& data, std::vector<float>& v, uint32_t n,
         uint32_t& nsb) {
        if (get_ptc_type(data.particles.tracked_data().flag[n]) ==
            (int)ParticleType::ion) {
          auto x = data.particles.tracked_data().x3[n];
          v[nsb] = x;
          nsb += 1;
        }
      },
      file, timestep);
}

}  // namespace Aperture

#endif  // __USER_DATA_OUTPUT_H_
