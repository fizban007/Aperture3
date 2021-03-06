#include "cuda/core/cu_sim_environment.h"
#include "cuda/data/array.h"
#include "radiation/spectra.h"
#include "sim.h"
#include "utils/logger.h"
#include "utils/timer.h"
#include "utils/util_functions.h"
#include <fstream>
#include <random>

#define H5_USE_BOOST
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>

using namespace Aperture;
using namespace HighFive;

int main(int argc, char *argv[]) {
  uint32_t step = 0;
  // Construct the simulation environment
  cu_sim_environment env(&argc, &argv);

  exponent_sim sim(env, false, true);

  // Spectra::black_body ne(1.685e-5);
  // Spectra::mono_energetic ne(1.0e-3, 1.0e-4);
  Spectra::power_law_soft ne(2.0, 0.2e-6, 0.2);
  // sim.init_spectra(ne, 1706.25);
  sim.init_spectra(ne, 1.75464e17 * 1706.25);
  // sim.init_spectra(ne, 1.0e18 * 1706.25);

  // sim.add_new_particles(100, 1.0);

  std::vector<Scalar> ic_rate(sim.m_ic.ic_rate().size());
  std::vector<Scalar> gg_rate(sim.m_ic.gg_rate().size());
  std::vector<Scalar> tpp_rate(sim.m_tpp.rate().size());

  for (int i = 0; i < ic_rate.size(); i++) {
    ic_rate[i] = sim.m_ic.ic_rate()[i];
  }
  for (int i = 0; i < gg_rate.size(); i++) {
    gg_rate[i] = sim.m_ic.gg_rate()[i];
  }
  for (int i = 0; i < tpp_rate.size(); i++) {
    tpp_rate[i] = sim.m_tpp.rate()[i];
  }
  {
    File datafile("rates.h5", File::ReadWrite | File::Create | File::Truncate);
    DataSet data_ic_rate =
        datafile.createDataSet<Scalar>("ic_rate", DataSpace::From(ic_rate));
    DataSet data_gg_rate =
        datafile.createDataSet<Scalar>("gg_rate", DataSpace::From(gg_rate));
    DataSet data_tpp_rate =
        datafile.createDataSet<Scalar>("tpp_rate", DataSpace::From(tpp_rate));
    data_ic_rate.write(ic_rate);
    data_gg_rate.write(gg_rate);
    data_tpp_rate.write(tpp_rate);
  }

  std::vector<Scalar> ptc_spec(sim.ptc_spec.size());
  std::vector<Scalar> ph_spec(sim.ph_spec.size());

  // std::vector<Scalar> Es{0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6,
  // 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0,
  // 400.0, 600.0, 800.0, 1000.0};
  std::vector<Scalar> Es{0.001,  0.003,  0.006,  0.01,    0.03,    0.06,
                         0.1,    0.3,    0.6,    1.0,     3.0,     6.0,
                         10.0,   30.0,   60.0,   100.0,   300.0,   600.0,
                         1000.0, 3000.0, 6000.0, 10000.0, 30000.0, 60000.0,
                         1.0e5,  3.0e5,  6.0e5,  1.0e6};

  // std::vector<Scalar> Es{10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0, 400.0,
  // 600.0, 800.0, 1000.0}; std::vector<Scalar> Es{0.0};
  std::vector<Scalar> exps;

  std::ofstream out;
  out.open("result_pl2.0.txt", std::ios::out | std::ios::trunc);

  for (auto Eacc : Es) {
    Scalar t_start, num_start, t_end;
    bool started = false;
    // double Eacc = env.params().constE;
    double dt = std::max(std::min(100.0 / Eacc, 2000.0), 100.0);
    // if (Eacc > 100.0)
    //   dt = 1000.0;
    double dt = env.params().delta_t;
    Logger::print_debug("dt is {}, Eacc is {}", dt, Eacc);
    // sim.prepare_initial_condition(100, 1.0);
    sim.prepare_initial_condition(100.0, 5.0e6);

    for (uint32_t step = 0; step < env.params().max_steps; step++) {
      // sim.add_new_particles(1, 1.0);
      sim.add_new_particles(100.0, 5.0e6);
      Logger::print_info(
          "==== On timestep {}, pushing {} particles, {} photons ====", step,
          sim.ptc_num, sim.ph_num);
      // sim.ptc_E.copy_to_host();
      // Logger::print_info("0th particle has energy {}", sim.ptc_E[0]);
      sim.push_particles(Eacc, dt);
      sim.produce_pairs(dt);
      sim.produce_photons(dt);
      sim.compute_spectrum();

      if (sim.ptc_num > 100000 && !started) {
        t_start = step * dt;
        num_start = sim.ptc_num;
        t_end = -0.01;
        started = true;
      }

      if (step % env.params().sort_interval == 0 && step != 0) {
        sim.sort_photons();
      }

      if (std::abs(Eacc - 0.0) < 1.0e-5 &&
          step % env.params().data_interval == 0) {
        File output_file(fmt::format("spec{:06d}.h5", step).c_str(),
                         File::ReadWrite | File::Create | File::Truncate);
        sim.ptc_spec.copy_to_host();
        sim.ph_spec.copy_to_host();
        for (unsigned int i = 0; i < ptc_spec.size(); i++) {
          ptc_spec[i] = sim.ptc_spec[i];
          ph_spec[i] = sim.ph_spec[i];
        }
        DataSet data_ptc_spec = output_file.createDataSet<Scalar>(
            "ptc_spec", DataSpace::From(ptc_spec));
        DataSet data_ph_spec = output_file.createDataSet<Scalar>(
            "ph_spec", DataSpace::From(ph_spec));
        data_ptc_spec.write(ptc_spec);
        data_ph_spec.write(ph_spec);
      }

      if (sim.ptc_num > 0.2 * env.params().max_ptc_number ||
          sim.ph_num > 0.5 * env.params().max_photon_number) {
        // || step >= 100000) {
        t_end = step * dt;
        break;
      }
    }
    if (t_end < 0.0)
      t_end = env.params().max_steps * dt;
    Scalar ex = (std::log(sim.ptc_num) - std::log(num_start)) /
                (t_end - t_start) / 5.6875e-8;
    Logger::print_info("Exponent for E = {}, kT = 10^5K is {}", Eacc, ex);
    Logger::print_debug("num_start = {}, num = {}, t_start = {}, t = {}",
                        num_start, sim.ptc_num, t_start, t_end);
    exps.push_back(ex);
    out << Eacc << ", " << ex << std::endl;
  }

  out.close();

  for (int i = 0; i < Es.size(); i++) {
    Logger::print_info("Exponent for E = {}, kT = 10^5K is {}", Es[i], exps[i]);
  }

  return 0;
}
