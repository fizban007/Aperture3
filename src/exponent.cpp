#include "cuda/constant_mem_func.h"
#include "cuda/cudarng.h"
#include "pic_sim.h"
#include "radiation/inverse_compton_power_law.h"
#include "radiation/radiation_transfer.h"
#include "sim_data_dev.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#include <fmt/core.h>
#include <fstream>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <iostream>
#include <random>
#include <vector>

using namespace Aperture;
using namespace HighFive;

template <typename Rad>
double
measure_exp(SimData& data, PICSim& sim, Environment& env, Rad& rad,
            double E) {
  Scalar dt = env.params().delta_t;
  env.params().constE = E;
  Logger::print_info("E is {}", env.params().constE);

  // Elastically scale dt to a semi-optimal level
  while (dt / env.params().ic_path < 1.0e-4) {
    Logger::print_info("dt is {}, characteristic ic rate is {}", dt,
                       dt / env.params().ic_path);
    dt *= 2.0;
  }
  while (E * dt > 0.1 / env.params().e_min ||
         dt / env.params().ic_path > 0.5) {
    Logger::print_info("dt is {}, characteristic ic rate is {}", dt,
                       dt / env.params().ic_path);
    dt *= 0.5;
  }

  Logger::print_info("dt is {}, characteristic ic rate is {}", dt,
                     dt / env.params().ic_path);
  env.params().delta_t = dt;
  init_dev_params(env.params());
  // Initialize seed particle(s)
  data.particles.initialize();
  data.photons.initialize();
  Logger::print_debug("Initializing particles");
  int N = 1000;
  // Add more seed particles for higher E field
  // if (E * env.params().e_min > 0.9) N *= 10;
  for (int i = 0; i < N; i++) {
    // data.particles.append({env.gen_rand(), 0.0, 0.0},
    // {-1.0f*env.gen_rand(), 0.0, 0.0}, 10, ParticleType::electron);
    data.particles.append({env.gen_rand(), 0.0, 0.0}, {-0.0f, 0.0, 0.0},
                          10, ParticleType::electron, 1.0,
                          bit_or(ParticleFlag::primary));
    // Logger::print_info("Particle flag is {}", data.particles.data().flag[0]);
  }
  // data.particles.sync_to_device();

  // output particle spectra
  std::vector<std::vector<uint32_t>> spectra_p, spectra_s, spectra_ph;
  std::vector<std::vector<Scalar>> ebins, ebins_ph;
  File spec_file(fmt::format("spec_emin{:.1e}E{:.0f}lic{:.1f}.h5",
                             env.params().e_min, env.params().constE,
                             env.params().ic_path),
                 File::ReadWrite | File::Create | File::Truncate);
  int n_bins = 256;

  // Main simulation loop
  Logger::print_debug("Starting simulation loop");
  size_t num_start = 0, num_end = 0;
  Scalar time_start = 0.0, time_end = 0.0;
  for (uint32_t step = 0; step < env.params().lph_cutoff / dt; step++) {
    // Logger::print_info("At timestep {}", step);
    // if (step % env.params().data_interval == 0) {

    // }
    if (step % 10 == 0) {
      data.particles.sort_by_cell();
      data.photons.sort_by_cell();
      Logger::print_info(
          "There are {} particles, {} photons in the pool",
          data.particles.number(), data.photons.number());
      Logger::print_info("p1 is at {}", data.particles.data().p1[0]);
      // TODO: Output particle spectrum here
      std::vector<uint32_t> sp;
      std::vector<Scalar> energies;
      data.particles.compute_energies();
      data.particles.compute_spectrum(n_bins, energies, sp, ParticleFlag::primary);
      // Logger::print_info("{}",energies[128]);
      spectra_p.push_back(sp);
      data.particles.compute_spectrum(n_bins, energies, sp, ParticleFlag::secondary);
      spectra_s.push_back(sp);
      ebins.push_back(energies);

      data.photons.compute_energies();
      data.photons.compute_spectrum(n_bins, energies, sp);
      spectra_ph.push_back(sp);
      ebins_ph.push_back(energies);
    }

    rad.emit_photons(data.photons, data.particles);
    // sim.ptc_pusher().push(data, dt);
    sim.ptc_updater().update_particles(data, dt);
    rad.produce_pairs(data.particles, data.photons);

    if (data.particles.number() > 20000 && num_start == 0) {
      num_start = data.particles.number();
      time_start = dt * step;
    }
    if (data.particles.number() > 40000000 ||
        data.photons.number() > 80000000 || step > 20000) {
      num_end = data.particles.number();
      time_end = dt * step;
      Logger::print_info("n_end = {}, n_start = {}", num_end,
                         num_start);
      Logger::print_info("t_end = {}, t_start = {}", time_end,
                         time_start);
      break;
    }
  }

  // write spectra to hdf5 file
  DataSet specset = spec_file.createDataSet<uint32_t>(
      "SpectraP", DataSpace::From(spectra_p));
  specset.write(spectra_p);
  DataSet specset2 = spec_file.createDataSet<uint32_t>(
      "SpectraS", DataSpace::From(spectra_s));
  specset2.write(spectra_s);
  DataSet specph = spec_file.createDataSet<uint32_t>(
      "SpectraPh", DataSpace::From(spectra_ph));
  specph.write(spectra_ph);
  DataSet Eset = spec_file.createDataSet<Scalar>(
      "Ebins", DataSpace::From(ebins));
  Eset.write(ebins);
  DataSet EsetPh = spec_file.createDataSet<Scalar>(
      "EbinsPh", DataSpace::From(ebins_ph));
  EsetPh.write(ebins_ph);

  double s = (std::log((double)num_end) - std::log((double)num_start)) /
             (time_end - time_start);
  Logger::print_info("At the end, exponent s = {}", s);
  return s;
}

int
main(int argc, char* argv[]) {
  Environment env(&argc, &argv);

  // Print the parameters of this run
  Logger::print_info("dt is {}", env.params().delta_t);
  // Logger::print_info("E is {}", env.params().constE);
  Logger::print_info("e_min is {}", env.params().e_min);

  // Allocate simulation data
  SimData data(env);

  // Initialize simulator
  PICSim sim(env);
  RadiationTransfer<Particles, Photons,
                    InverseComptonPL1D<Kernels::CudaRng>>
      rad(env);

  std::vector<double> Es;
  // = {50.0, 1.0e2, 5.0e2, 1.0e3, 5.0e3, 1.0e4, 5.0e4,
  //                           1.0e5, 5.0e5, 1.0e6, 5.0e6, 1.0e7};
  for (double E = 0.1 / env.params().e_min;
       E < 100.0 / env.params().e_min; E *= 2.0) {
    Es.push_back(E);
  }
  std::vector<double> ss(Es.size(), 0.0);

  std::string fname =
      "emin" + fmt::format("{:.1e}", env.params().e_min) + "lic" +
      fmt::format("{:.1f}", env.params().ic_path) + "alpha" +
      fmt::format("{:.1f}.txt", env.params().spectral_alpha);
  // fmt::print(Logger::m_file, "Test");
  std::ofstream outfile;
  outfile.open(fname);
  outfile << "For emin = " << env.params().e_min << std::endl;
  outfile << "l_IC = " << env.params().ic_path
          << ", alpha = " << env.params().spectral_alpha << std::endl;

  Scalar dt = env.params().delta_t;
  for (unsigned int i = 0; i < Es.size(); i++) {
    ss[i] = measure_exp(data, sim, env, rad, Es[i]);
    // Logger::log_info("E = {}, s = {}", Es[i], ss[i]);
    outfile << "E = " << Es[i] << ", s = " << ss[i]
            << ", dt = " << env.params().delta_t << std::endl;
    env.params().delta_t = dt;
  }

  outfile.close();

  return 0;
}
