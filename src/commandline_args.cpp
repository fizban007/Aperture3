#include "commandline_args.h"
#include "cxxopts.hpp"
#include "sim_params.h"
#include "utils/logger.h"
#include <cstdint>
#include <iostream>

namespace Aperture {

CommandArgs::CommandArgs() {
  SimParams defaults;

  m_options = std::unique_ptr<cxxopts::Options>(
      new cxxopts::Options("aperture", "Aperture PIC code"));

  m_options->add_options()("h,help", "Prints this help message.")(
      "c,config", "Configuration file for the simulation.",
      cxxopts::value<std::string>()->default_value(defaults.conf_file))(
      // "s,steps", "Number of steps to run the simulation.",
      // cxxopts::value<uint32_t>()->default_value(
      //     std::to_string(defaults.max_steps)))(
      // "d,interval", "The interval to output data to the hard disk.",
      // cxxopts::value<uint32_t>()->default_value(
      //     std::to_string(defaults.data_interval)))(
      // "x,dimx", "The number of processes in x direction.",
      // cxxopts::value<int>()->default_value("1"))(
      // "y,dimy", "The number of processes in y direction.",
      // cxxopts::value<int>()->default_value("1"))(
      // "z,dimz", "The number of processes in z direction.",
      // cxxopts::value<int>()->default_value("1"))(
      "r,restart_file", "The restart file used in this run.",
      cxxopts::value<std::string>()->default_value(""));
}

CommandArgs::~CommandArgs() {}

void
CommandArgs::read_args(int argc, char* argv[], SimParams& params) {
  try {
    auto result = m_options->parse(argc, argv);

    if (result["help"].as<bool>()) {
      std::cout << m_options->help() << std::endl;
      exit(0);
    }
    // params.max_steps = result["steps"].as<uint32_t>();
    // params.data_interval = result["interval"].as<uint32_t>();
    // params.dim_x = result["dimx"].as<int>();
    // params.dim_y = result["dimy"].as<int>();
    // params.dim_z = result["dimz"].as<int>();
    params.conf_file = result["config"].as<std::string>();
    params.restart_file = result["restart_file"].as<std::string>();
  } catch (std::exception& e) {
    Logger::err("Error: {}", e.what());
    // Logger::err(e.what());
    std::cout << m_options->help() << std::endl;
    throw(exceptions::program_option_terminate());
  }
}
}  // namespace Aperture
