#include <iostream>
#include <cstdint>
#include "commandline_args.h"
#include "utils/logger.h"
#include "cxxopts.hpp"

// using namespace Aperture;
namespace Aperture {
// namespace po = boost::program_options;

CommandArgs::CommandArgs() {
  m_options = std::make_unique<cxxopts::Options>("aperture", "Aperture PIC code");
  m_options->add_options()
      ("h,help", "Prints this help message.")
      // ("verbose,v", po::value<int>(&verbosity)->default_value(0)->implicit_value(3),
      //  "Level of verbosity of program output.")
      // ("config,c", po::value<std::string>(&m_conf_filename)->default_value("sim.conf"),
       // "Configuration file for the simulation.")
      ("c,config",
       "Configuration file for the simulation.", cxxopts::value<std::string>()->default_value("sim.conf"))
      ("s,steps",
       "Number of steps to run the simulation.", cxxopts::value<uint32_t>()->default_value("2000"))
      ("d,interval",
       "The interval to output data to the hard disk.", cxxopts::value<uint32_t>()->default_value("20"))
      // ("mode,m", po::value<std::string>(&mode)->default_value("cpu"),
      //  "Execution mode, can be either cpu or gpu.")
      ("x,dimx",
       "The number of processes in x direction.", cxxopts::value<int>()->default_value("1"))
      ("y,dimy",
       "The number of processes in y direction.", cxxopts::value<int>()->default_value("1"))
      ("z,dimz",
       "The number of processes in z direction.", cxxopts::value<int>()->default_value("1"));
}

CommandArgs::~CommandArgs() {}

void
CommandArgs::read_args(int argc, char* argv[]) {
  try {
    Logger::print_info("Reading arguments");
    auto result = m_options->parse(argc, argv);
    // po::variables_map vm;
    // po::store(po::parse_command_line(argc, argv, _desc), vm);
    // po::notify(vm);

    // if (result.count("help")) {
    if (result["help"].as<bool>()) {
      Logger::print_info("Display help");
      // std::cout << _desc << std::endl;
      std::cout << m_options->help({""}) << std::endl;
      // throw(exceptions::program_option_terminate());
      exit(0);
    }
    m_steps = result["steps"].as<uint32_t>();
    m_data_interval = result["interval"].as<uint32_t>();
    m_conf_filename = result["config"].as<std::string>();
    m_dimx = result["x"].as<int>();
    m_dimy = result["y"].as<int>();
    m_dimz = result["z"].as<int>();
  } catch (std::exception& e) {
    Logger::err("Error");
    Logger::err(e.what());
    std::cout << m_options->help({""}) << std::endl;
    // std::cout << _desc << std::endl;
    throw(exceptions::program_option_terminate());
  }
}
}
