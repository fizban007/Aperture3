#ifndef __SIM_ENVIRONMENT_IMPL_H_
#define __SIM_ENVIRONMENT_IMPL_H_

#include "grids/grid_log_sph.h"
#include "grids/grid_1dgr.h"
#include "sim_data.h"
#include "sim_environment.h"
#include "utils/logger.h"
#include <boost/filesystem.hpp>
#include <memory>
// #include "domain_communicator.h"

namespace Aperture {


sim_environment::sim_environment(int* argc, char*** argv)
    : m_generator(), m_dist(0.0, 1.0) {
  int is_initialized = 0;
  MPI_Initialized(&is_initialized);

  if (!is_initialized) {
    if (argc == nullptr && argv == nullptr) {
      MPI_Init(NULL, NULL);
    } else {
      MPI_Init(argc, argv);
    }
  }

  m_world = MPI_COMM_WORLD;
  MPI_Comm_rank(m_world, &m_domain_info.rank);
  MPI_Comm_size(m_world, &m_domain_info.size);
  // Initialize logger for future use
  Logger::init(m_domain_info.rank, m_params.log_lvl, m_params.log_file);

  Logger::print_info("Num of ranks is {}", m_domain_info.size);
  Logger::print_info("Current rank is {}", m_domain_info.rank);

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
  m_charges[(int)ParticleType::electron] *= -1.0;
  m_masses[(int)ParticleType::ion] *= m_params.ion_mass;
  Logger::print_debug("Electron particle type is {}, charge_e is {}",
                      (int)ParticleType::electron, m_charges[0]);
  for (int i = 0; i < 8; i++) {
    m_q_over_m[i] = m_charges[i] / m_masses[i];
  }

  // Setup the domain
  setup_domain(m_domain_info.size);

  // Setup the local grid and the local data output grid
  setup_local_grid();

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

  // Split the whole world into number of cartesian dimensions
  Logger::print_info("nodes from params file is {}x{}x{}", m_params.nodes[0],
                     m_params.nodes[1], m_params.nodes[2]);
  int dims[3] = {1, 1, 1};
  int total_dim = 1;
  for (int i = 0; i < 3; i++) {
    dims[i] = m_params.nodes[i];
    total_dim *= dims[i];
  }

  if (total_dim != m_domain_info.size) {
    // Given node configuration is not correct, create one on our own
    std::cerr << "Domain decomp in config file does not make sense!"
              << std::endl;
    for (int i = 0; i < 3; i++) dims[i] = 0;
    MPI_Dims_create(m_domain_info.size, m_super_grid->dim(), dims);
  }

  if (ndims == 1) setup_domain(dims[0], 1);
  if (ndims == 2) setup_domain(dims[0], dims[1]);
  if (ndims == 3) setup_domain(dims[0], dims[1], dims[2]);
}

void
sim_environment::setup_domain(int dimx, int dimy, int dimz) {
  // First create Cartesian rank group
  for (int i = 0; i < 3; i++) {
    m_domain_info.is_periodic[i] = m_params.periodic_boundary[i];
  }
  int dims[3]{dimx, dimy, dimz};

  for (int i = 0; i < 3; i++) m_domain_info.cart_dims[i] = dims[i];

  // Create a cartesian MPI group for communication
  MPI_Cart_create(m_world, 3, dims, m_domain_info.is_periodic, true, &m_cart);

  // Obtain the mpi coordinate of the current rank
  MPI_Cart_coords(m_cart, m_domain_info.rank, 3, m_domain_info.cart_coord);

  Logger::print_info("Domain created with dimensions {} x {} x {}",
                      dimx, dimy, dimz);
  Logger::print_info("Rank {} has coords {}, {}, {}",
                      m_domain_info.rank, m_domain_info.cart_coord[0],
                      m_domain_info.cart_coord[1], m_domain_info.cart_coord[2]);

  // Initialize domain_info
  // m_domain_info.rank_map.resize(dimz);

  // Process neighbor rank information
  // Figure out if the current rank is at any boundary
  int xleft, xright, yleft, yright, zleft, zright;
  int rank;
  MPI_Cart_shift(m_cart, 0, -1, &rank, &xleft);
  MPI_Cart_shift(m_cart, 0, 1, &rank, &xright);
  m_domain_info.neighbor_left[0] = xleft;
  m_domain_info.neighbor_right[0] = xright;
  if (xleft < 0) m_domain_info.is_boundary[0] = true;
  if (xright < 0) m_domain_info.is_boundary[1] = true;

  MPI_Cart_shift(m_cart, 1, -1, &rank, &yleft);
  MPI_Cart_shift(m_cart, 1, 1, &rank, &yright);
  m_domain_info.neighbor_left[1] = yleft;
  m_domain_info.neighbor_right[1] = yright;
  if (yleft < 0) m_domain_info.is_boundary[2] = true;
  if (yright < 0) m_domain_info.is_boundary[3] = true;

  MPI_Cart_shift(m_cart, 2, -1, &rank, &zleft);
  MPI_Cart_shift(m_cart, 2, 1, &rank, &zright);
  m_domain_info.neighbor_left[2] = zleft;
  m_domain_info.neighbor_right[2] = zright;
  if (zleft < 0) m_domain_info.is_boundary[4] = true;
  if (zright < 0) m_domain_info.is_boundary[5] = true;

}

void
sim_environment::setup_local_grid() {
  if (m_params.coord_system == "LogSpherical") {
    m_grid.reset(new Grid_LogSph());
  } else if (m_params.coord_system == "1DGR" && m_super_grid->dim() == 1) {
    m_grid.reset(new Grid_1dGR());
  } else {
    m_grid.reset(new Grid());
  }
  m_grid->init(m_params);
  auto& mesh = m_grid->mesh();
  for (int d = 0; d < 3; d++) {
    if (m_domain_info.cart_dims[d] > 1) {
      // We adjust local params, and use these params to init the grid
      // m_params.N[d] /= m_domain_info.cart_dims[d];
      // m_params.size[d] /= m_domain_info.cart_dims[d];
      // m_params.lower[d] += m_domain_info.cart_pos[d] *
      // m_params.size[d];
      mesh.dims[d] = mesh.reduced_dim(d) / m_domain_info.cart_dims[d] +
          2 * mesh.guard[d];
      mesh.sizes[d] /= m_domain_info.cart_dims[d];
      mesh.lower[d] += m_domain_info.cart_coord[d] * mesh.sizes[d];
      mesh.offset[d] = m_domain_info.cart_coord[d] * mesh.reduced_dim(d);
      Logger::print_info("offset[{}] is {}", d, mesh.offset[d]);
    }
  }
  m_grid->compute_coef(m_params);

  m_send_buffers.resize(3);
  m_recv_buffers.resize(3);
  m_send_buffers[0] = multi_array<Scalar>(
      mesh.guard[0], mesh.dims[1], mesh.dims[2]);
  m_recv_buffers[0] = multi_array<Scalar>(
      mesh.guard[0], mesh.dims[1], mesh.dims[2]);
  if (mesh.dim() > 1) {
    m_send_buffers[1] =
        multi_array<Scalar>(mesh.dims[0], mesh.guard[1], mesh.dims[2]);
    m_recv_buffers[1] =
        multi_array<Scalar>(mesh.dims[0], mesh.guard[1], mesh.dims[2]);
  }
  if (mesh.dim() > 2) {
    m_send_buffers[2] =
        multi_array<Scalar>(mesh.dims[0], mesh.dims[1], mesh.guard[2]);
    m_recv_buffers[2] =
        multi_array<Scalar>(mesh.dims[0], mesh.dims[1], mesh.guard[2]);
  }
}

void
sim_environment::send_array_guard_cells(multi_array<Scalar> &array) {
  send_array_guard_cells_x(array, -1);
  send_array_guard_cells_x(array, 1);
  if (m_grid->dim() >= 2) {
    send_array_guard_cells_y(array, -1);
    send_array_guard_cells_y(array, 1);
  }
  if (m_grid->dim() >= 3) {
    send_array_guard_cells_z(array, -1);
    send_array_guard_cells_z(array, 1);
  }
}

void
sim_environment::send_add_array_guard_cells(multi_array<Scalar> &array) {
  send_add_array_guard_cells_x(array, -1);
  send_add_array_guard_cells_x(array, 1);
  if (m_grid->dim() >= 2) {
    send_add_array_guard_cells_y(array, -1);
    send_add_array_guard_cells_y(array, 1);
  }
  if (m_grid->dim() >= 3) {
    send_add_array_guard_cells_z(array, -1);
    send_add_array_guard_cells_z(array, 1);
  }
}

}  // namespace Aperture

#endif  // __SIM_ENVIRONMENT_IMPL_H_
