#include "sim_environment.h"
#include "fmt/format.h"
#include "grids/grid_log_sph.h"
#include "sim_data.h"
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
    exit(0);
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
  for (int i = 0; i < 8; i++) {
    m_q_over_m[i] = m_charges[i] / m_masses[i];
  }

  // Setup the domain
  setup_domain(m_comm->world().size());

  // Setup the local grid and the local data output grid
  setup_local_grid();

  // Now we need to decide what to initiate depending on what is the
  // user configuration. Need to discuss this and think it through.
  // Logger::print_debug("Local mesh is {}", m_local_grid.mesh());

  // call setup_metric according to the selected metric on the local
  // grid of each node, setting up the metric coefficients and other
  // cache arrays for the local grid select_metric(m_metric_type,
  // m_local_grid.setup_metric, m_local_grid);

  // initialize the data exporter
  // m_exporter = std::make_unique<DataExporter>(
  //     m_params.data_dir, m_params.data_file_prefix);
  // m_exporter = std::unique_ptr<DataExporter>(new DataExporter(
  //     m_grid, m_params.data_dir, m_params.data_file_prefix));

  // Initialize logger for future use
  // Logger::init(m_comm->world().rank(), m_conf_file.data().log_lvl,
  // m_conf_file.data().log_file);
  Logger::init(0, m_params.log_lvl, m_params.log_file);
  // Logger::print_debug("Current rank is {}", m_comm->world().rank());

  Logger::print_info("Setup environment completed.");
  // Logger::print_info("Each particle is worth {} bytes",
  //                    particle_data::size);
  // Logger::print_info("Each photon is worth {} bytes",
  //                    photon_data::size);
}

void
sim_environment::setup_domain(int num_nodes) {
  int ndims = m_grid->dim();

  // Start to create dims from scratch
  int dims[3] = {1, 1, 1};
  for (int i = 0; i < ndims; ++i) dims[i] = 0;

  m_comm->cartesian().create_dims(num_nodes, ndims, dims);

  if (ndims == 1) setup_domain(dims[0]);
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

  // Cache whether this node is boundary on all 6 directions
  // for (int i = 0; i < NUM_BOUNDARIES; ++i)
  //   m_domain_info.is_boundary[i] = false;
  // if (m_domain_info.cart_neighbor_left[0] == NEIGHBOR_NULL)
  //   m_domain_info.is_boundary[0] = true;
  // if (m_domain_info.cart_neighbor_right[0] == NEIGHBOR_NULL)
  //   m_domain_info.is_boundary[1] = true;
  // if (m_domain_info.cart_neighbor_left[1] == NEIGHBOR_NULL)
  //   m_domain_info.is_boundary[2] = true;
  // if (m_domain_info.cart_neighbor_right[1] == NEIGHBOR_NULL)
  //   m_domain_info.is_boundary[3] = true;

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

// void
// sim_environment::set_initial_condition(InitialCondition* ic) {
//   if (m_ic == nullptr)
//     m_ic.reset(ic);
//   else
//     Logger::print_err("Initial condition already set!");
// }

// void
// sim_environment::set_initial_condition(cu_sim_data& data, const
// Index& start, const Extent& extent) {}

// void
// sim_environment::add_fieldBC(fieldBC *bc) {
//   // Only add boundary condition if the node is at boundary
//   int pos = static_cast<int>(bc->pos());
//   if (m_domain_info.is_boundary[pos]) {
//     Logger::print_debug_all("Rank {} Adding boundary condition at pos
//     {}", m_domain_info.rank, pos); m_bc.add_fieldBC(bc);
//   }// }

// void
// sim_environment::add_ptcBC(ptcBC* bc) {
//   m_bc.add_ptcBC(bc);
// }

void
sim_environment::setup_local_grid() {
  for (int d = 0; d < 3; d++) {
    if (m_domain_info.cart_dims[d] > 0) {
      // local_mesh.dims[d] = super_mesh.reduced_dim(d) /
      // info.cart_dims[d] +
      //                     2 * super_mesh.guard[d];
      // local_mesh.sizes[d] /= info.cart_dims[d];
      // local_mesh.lower[d] += info.cart_pos[d] *
      // local_mesh.reduced_dim(d) * local_mesh.delta[d];

      // We adjust local params, and use these params to init the grid
      m_params.N[d] /= m_domain_info.cart_dims[d];
      m_params.size[d] /= m_domain_info.cart_dims[d];
      m_params.lower[d] += m_domain_info.cart_pos[d] * m_params.size[d];
    }
  }
  m_grid.reset(new Grid());
  m_grid->init(m_params);
}

// void
// sim_environment::apply_initial_condition(cu_sim_data &data) {
//   Logger::print_info("Applying initial condition");
//   if (m_ic == nullptr) {
//     Logger::print_err("No initial condition set yet!");
//     return;
//   }

//   data.E.initialize([this, &data](int n, Scalar x1, Scalar x2,
//   Scalar x3){
//       return this->initial_condition().E(n, x1, x2, x3,
//       data.E.grid().mesh());
//     });

//   data.B.initialize([this, &data](int n, Scalar x1, Scalar x2,
//   Scalar x3){
//       return this->initial_condition().B(n, x1, x2, x3,
//       data.B.grid().mesh());
//     });

//   m_bc.apply_fieldBC(data.E, data.B);

//   m_ic->set_data(data);
//   m_bc.initialize(*this, data);
// }

}  // namespace Aperture
