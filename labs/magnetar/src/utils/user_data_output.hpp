#ifndef __USER_DATA_OUTPUT_H_
#define __USER_DATA_OUTPUT_H_

#include "sim_data.h"
#include "sim_environment.h"
#include "utils/data_exporter.h"
#include "utils/util_functions.h"

namespace Aperture {

void
user_write_field_output(sim_data& data, data_exporter& exporter,
                        uint32_t timestep, double time, H5File& file) {
  exporter.add_grid_output(data.E.data(0), data.E.stagger(0), "E1", file,
                  timestep);
  exporter.add_grid_output(data.E.data(1), data.E.stagger(1), "E2", file,
                  timestep);
  exporter.add_grid_output(data.E.data(2), data.E.stagger(2), "E3", file,
                  timestep);
  exporter.add_grid_output(data.B.data(0), data.B.stagger(0), "B1", file,
                  timestep);
  exporter.add_grid_output(data.B.data(1), data.B.stagger(1), "B2", file,
                  timestep);
  exporter.add_grid_output(data.B.data(2), data.B.stagger(2), "B3", file,
                  timestep);
  exporter.add_grid_output(data.J.data(0), data.J.stagger(0), "J1", file,
                  timestep);
  exporter.add_grid_output(data.J.data(1), data.J.stagger(1), "J2", file,
                  timestep);
  exporter.add_grid_output(data.J.data(2), data.J.stagger(2), "J3", file,
                  timestep);
  exporter.add_grid_output(data.Rho[0].data(), data.Rho[0].stagger(), "Rho_e",
                  file, timestep);
  exporter.add_grid_output(data.Rho[1].data(), data.Rho[1].stagger(), "Rho_p",
                  file, timestep);
  if (data.env.params().num_species > 2) {
    exporter.add_grid_output(data.Rho[2].data(), data.Rho[2].stagger(), "Rho_i",
                    file, timestep);
  }
  exporter.add_grid_output(data.divE.data(), data.divE.stagger(), "divE",
                  file, timestep);
  exporter.add_grid_output(data.divB.data(), data.divB.stagger(), "divB",
                  file, timestep);
  exporter.add_grid_output(data.photon_produced.data(), data.photon_produced.stagger(), "photon_produced",
                  file, timestep);
  exporter.add_grid_output(data.pair_produced.data(), data.pair_produced.stagger(), "pair_produced",
                  file, timestep);
  // exporter.add_array_output(data.ph_flux, "ph_flux", file, timestep);
  file.write(data.ph_flux, "ph_flux");
}

void
user_write_ptc_output(sim_data& data, data_exporter& exporter,
                      uint32_t timestep, double time, H5File& file) {
}

}  // namespace Aperture

#endif  // __USER_DATA_OUTPUT_H_
