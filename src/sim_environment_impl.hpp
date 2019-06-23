#ifndef __SIM_ENVIRONMENT_IMPL_H_
#define __SIM_ENVIRONMENT_IMPL_H_

#include "grids/grid_log_sph.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include <boost/filesystem.hpp>
#include <memory>
// #include "domain_communicator.h"

namespace Aperture {

sim_environment::sim_environment(int* argc, char*** argv)
    : m_generator(), m_dist(0.0, 1.0) {
  m_comm = std::make_unique<MPIComm>(argc, argv);

  // Read in command line configuration
  // Handle the case of wrong command line arguments, exit gracefully
  try {
    m_args.read_args(*argc, *argv, m_params);
  } catch (exceptions::program_option_terminate& e) {
    exit(0);
  }

  // Read in the input file
  try {
    m_conf_file.parse_file(m_params.conf_file, m_params);
  } catch (exceptions::file_not_found& e) {
    Logger::err("Config file not found, exiting");
    exit(1);
  }

  // Look at the output directory to see if we are restarting from a
  // snapshot
  boost::filesystem::path snapshotPath(m_params.data_dir);
  snapshotPath /= "snapshot.h5";
  Logger::print_info("Snapshot path is {}", snapshotPath.string());
  boost::filesystem::path config_path(m_params.data_dir);
  config_path /= "config.toml";
  if (boost::filesystem::exists(snapshotPath) &&
      boost::filesystem::exists(config_path)) {
    // Reading from a snapshot, use the config file in data output path
    // instead
    Logger::print_info(
        "**** Found a snapshot file, reading its config instead!");
    m_params.conf_file = config_path.string();
    m_conf_file.parse_file(m_params.conf_file, m_params);
    m_params.is_restart = true;
  }

  setup_env();
}

sim_environment::sim_environment(const std::string& conf_file)
    : m_generator(), m_dist(0.0, 1.0) {
  m_params.conf_file = conf_file;
  // Read in the input file
  try {
    m_conf_file.parse_file(m_params.conf_file, m_params);
  } catch (exceptions::file_not_found& e) {
    Logger::err("Config file not found, exiting");
    exit(0);
  }

  setup_env();
}

sim_environment::~sim_environment() {}

void
sim_environment::setup_env() {
  // Init a dummy grid to store the geometry of the overall grid for the
  // simulation
  m_super_grid.reset(new Grid());
  m_super_grid->init(m_params);

  // Setup particle charges and masses
  for (int i = 0; i < 8; i++) {
    m_charges[i] = m_params.q_e;
    m_masses[i] = m_params.q_e;
  }
  Logger::print_debug("Electron particle type is {}, charge_e is {}",
                      (int)ParticleType::electron, m_charges[0]);
  m_charges[(int)ParticleType::electron] *= -1.0;
  m_masses[(int)ParticleType::ion] *= m_params.ion_mass;
  for (int i = 0; i < 8; i++) {
    m_q_over_m[i] = m_charges[i] / m_masses[i];
  }

  // Setup the domain
  setup_domain(m_comm->world().size());

  // Setup the local grid and the local data output grid
  setup_local_grid();

  // Initialize logger for future use
  Logger::init(0, m_params.log_lvl, m_params.log_file);
  Logger::print_debug("Current rank is {}", m_comm->world().rank());

  Logger::print_info("Setup environment completed.");
  Logger::print_info("Each particle is worth {} bytes",
                     particle_data::size);
  Logger::print_info("Each photon is worth {} bytes",
                     photon_data::size);

  setup_env_extra();
}

void
sim_environment::setup_domain(int num_nodes) {
  int ndims = m_super_grid->dim();

  // Start to create dims from scratch
  int dims[3] = {1, 1, 1};
  for (int i = 0; i < ndims; ++i) dims[i] = 0;

  m_comm->cartesian().create_dims(num_nodes, ndims, dims);

  if (ndims == 1) setup_domain(dims[0], 1);
  if (ndims == 2) setup_domain(dims[0], dims[1]);
  if (ndims == 3) setup_domain(dims[0], dims[1], dims[2]);
}

void
sim_environment::setup_domain(int dimx, int dimy, int dimz) {
  // First create Cartesian rank group
  bool periodic[3];
  for (int i = 0; i < 3; i++) {
    periodic[i] = m_params.periodic_boundary[i];
  }
  int dims[3]{dimx, dimy, dimz};
  auto cartesian_ranks =
      m_comm->get_cartesian_members(dimx * dimy * dimz);
  m_comm->cartesian().create_cart(m_super_grid->dim(), dims, periodic,
                                  cartesian_ranks);
  //   // if (!m_comm->cartesian().is_null())
  //   //   m_domain_info.state = ProcessState::primary;

  Logger::print_debug("Domain created with dimensions {} x {} x {}",
                      dimx, dimy, dimz);

  // Initialize domain_info
  m_domain_info.dim = m_super_grid->dim();
  // m_domain_info.rank_map.resize(dimz);
  m_domain_info.rank = m_comm->world().rank();
  m_domain_info.cart_pos.x = m_comm->cartesian().coord(0);
  if (m_domain_info.dim >= 2)
    m_domain_info.cart_pos.y = m_comm->cartesian().coord(1);
  if (m_domain_info.dim >= 3)
    m_domain_info.cart_pos.z = m_comm->cartesian().coord(2);

  // Process neighbor rank information
  for (int i = 0; i < 3; i++) {
    m_domain_info.cart_dims[i] = dims[i];
    if (i < m_domain_info.dim) {
      m_domain_info.cart_neighbor_left[i] =
          m_comm->cartesian().neighbor_left(i);
      Logger::print_debug_all(
          "On rank {}, left neighbor in dir {} is({}) ",
          m_domain_info.rank, i, m_domain_info.cart_neighbor_left[i]);
      if (m_domain_info.cart_neighbor_left[i] == NEIGHBOR_NULL)
        m_domain_info.is_boundary[i * 2] = true;

      m_domain_info.cart_neighbor_right[i] =
          m_comm->cartesian().neighbor_right(i);
      if (m_domain_info.cart_neighbor_right[i] == NEIGHBOR_NULL)
        m_domain_info.is_boundary[i * 2 + 1] = true;
      Logger::print_debug_all(
          "On rank {}, right neighbor in dir {} is ({})",
          m_domain_info.rank, i, m_domain_info.cart_neighbor_right[i]);
    } else {
      m_domain_info.cart_neighbor_left[i] =
          m_domain_info.cart_neighbor_right[i] = NEIGHBOR_NULL;
      m_domain_info.is_boundary[i * 2] =
          m_domain_info.is_boundary[i * 2 + 1] = true;
    }
  }

  //   // Debug info for boundary
  //   // Logger::print_debug_all("Rank {} has boundary info {} {} {}
  //   {}", m_domain_info.rank, m_domain_info.is_boundary[0],
  //   m_domain_info.is_boundary[1], m_domain_info.is_boundary[2],
  //   m_domain_info.is_boundary[3]);

  //   // resize domain map
  //   // for (auto& v : m_domain_info.rank_map) {
  //   //   v.resize(dimy);
  //   //   for (auto& u : v) {
  //   //     u.resize(dimx, 0);
  //   //   }
  //   // }

  //   // Populating the domain map
  //   // int *domain_pos_x, *domain_pos_y, *domain_pos_z;
  //   // domain_pos_x = new int[m_comm->cartesian().size()];
  //   // domain_pos_y = new int[m_comm->cartesian().size()];
  //   // domain_pos_z = new int[m_comm->cartesian().size()];
  //   // // domain_pos_x[m_rank] = m_comm -> coord(0);
  //   // // domain_pos_y[m_rank] = m_comm -> coord(1);
  //   // // domain_pos_z[m_rank] = m_comm -> coord(2);
  //   // m_comm->cartesian().all_gather(&m_domain_info.cart_pos.x, 1,
  //   domain_pos_x,
  //   //                                 1);
  //   // m_comm->cartesian().all_gather(&m_domain_info.cart_pos.y, 1,
  //   domain_pos_y,
  //   //                                 1);
  //   // m_comm->cartesian().all_gather(&m_domain_info.cart_pos.z, 1,
  //   domain_pos_z,
  //   //                                 1);
  //   // for (int i = 0; i < m_comm->cartesian().size(); i++) {
  //   //
  //   m_domain_info.rank_map[domain_pos_z[i]][domain_pos_y[i]][domain_pos_x[i]]
  //   //   =
  //   //       i;
  //   // }

  //   // delete[] domain_pos_x;
  //   // delete[] domain_pos_y;
  //   // delete[] domain_pos_z;
}

void
sim_environment::setup_local_grid() {
  if (m_params.coord_system == "LogSpherical") {
    m_grid.reset(new Grid_LogSph());
    // } else if (m_params.coord_system == "1DGR" &&
    //            m_grid->dim() == 1) {
    //   m_grid.reset(new Grid_1dGR_dev());
  } else {
    m_grid.reset(new Grid());
  }
  m_grid->init(m_params);
  auto& mesh = m_grid->mesh();
  for (int d = 0; d < 3; d++) {
    if (m_domain_info.cart_dims[d] > 0) {
      // We adjust local params, and use these params to init the grid
      // m_params.N[d] /= m_domain_info.cart_dims[d];
      // m_params.size[d] /= m_domain_info.cart_dims[d];
      // m_params.lower[d] += m_domain_info.cart_pos[d] *
      // m_params.size[d];
      mesh.dims[d] = mesh.reduced_dim(d) / m_domain_info.cart_dims[d] +
          2 * mesh.guard[d];
      mesh.sizes[d] /= m_domain_info.cart_dims[d];
      mesh.lower[d] += m_domain_info.cart_pos[d] * mesh.sizes[d];
      mesh.offset[d] = m_domain_info.cart_pos[d] * mesh.reduced_dim(d);
    }
  }
  m_grid->compute_coef(m_params);
}

}  // namespace Aperture

#endif  // __SIM_ENVIRONMENT_IMPL_H_
