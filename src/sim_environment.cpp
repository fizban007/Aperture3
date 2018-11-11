#include "sim_environment.h"
#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cudaUtility.h"
#include "fmt/format.h"
#include <memory>
#include "data/grid_log_sph.h"
#include "data/grid_1dGR.h"
// #include "data/detail/grid_impl.hpp"
#include "sim_data.h"
// #include "domain_communicator.h"

namespace Aperture {

// Environment&
Environment::Environment(int* argc, char*** argv)
    : m_generator(), m_dist(0.0, 1.0) {
  // m_comm = std::make_unique<MPIComm>(argc, argv);
  // m_comm = std::make_unique<MPIComm>(nullptr, nullptr);

  // Read in command line configuration
  // Handle the case of wrong command line arguments, exit gracefully
  try {
    m_args.read_args(*argc, *argv, m_params);
  } catch (exceptions::program_option_terminate& e) {
    exit(0);
  }

  setup_env(m_params.conf_file);
}

Environment::Environment(const std::string& conf_file)
    : m_generator(), m_dist(0.0, 1.0) {
  setup_env(conf_file);
}

Environment::~Environment() {}

void
Environment::setup_env(const std::string& conf_file) {
  m_params.conf_file = conf_file;
  // Read in the input file
  try {
    m_conf_file.parse_file(conf_file, m_params);
  } catch (exceptions::file_not_found& e) {
    Logger::err("Config file not found, exiting");
    exit(0);
  }

  // SimParamsBase* h_params = &m_params;

  // Copy the parameters to cuda constant memory
  // cudaMemcpyToSymbol(dev_params, (void*)&m_params,
  // sizeof(SimParamsBase)); CudaCheckError();
  init_dev_params(m_params);

  // Setup the grid
  if (m_params.coord_system == "Cartesian") {
    // m_grid = std::make_shared<Grid>();
    m_grid.reset(new Grid());
  } else if (m_params.coord_system == "LogSpherical") {
    // m_grid = std::make_shared<Grid_LogSph>();
    m_grid.reset(new Grid_LogSph());
  } else {
    // m_grid = std::make_shared<Grid>();
    m_grid.reset(new Grid());
  }
  m_grid->init(m_params);
  std::cout << "Grid dimension is " << m_grid->dim() << std::endl;
  if (m_grid->mesh().delta[0] < m_params.delta_t) {
    std::cerr << "Grid spacing should be larger than delta_t! Aborting!"
              << std::endl;
    abort();
  }
  // std::cout << m_grid.mesh().dims[0] << ", " << m_grid.mesh().dims[1]
  // << std::endl; std::cout << "size of quadmesh is " <<
  // sizeof(Quadmesh) << std::endl; cudaMemcpyToSymbol(dev_mesh,
  // (void*)m_grid.mesh_ptr(), sizeof(Quadmesh)); CudaCheckError();
  init_dev_mesh(*(m_grid->mesh_ptr()));

  // Initialize the background fields
  if (m_params.use_bg_fields) {
    m_Ebg = VectorField<Scalar>(*m_grid);
    m_Bbg = VectorField<Scalar>(*m_grid);
    init_dev_bg_fields(m_Ebg, m_Bbg);
  }

  // Setup particle charges and masses
  float charges[8];
  float masses[8];
  for (int i = 0; i < 8; i++) {
    charges[i] = m_params.q_e;
    masses[i] = m_params.q_e;
  }
  charges[(int)ParticleType::electron] *= -1.0;
  masses[(int)ParticleType::ion] *= m_params.ion_mass;
  init_dev_charges(charges);
  init_dev_masses(masses);
  // cudaMemcpyToSymbol(dev_charges, (void*)charges, sizeof(charges));
  // cudaMemcpyToSymbol(dev_masses, (void*)masses, sizeof(masses));
  // CudaCheckError();

  // Obtain the metric type and setup the grid mesh
  // m_metric_type = parse_metric(m_conf_file.data().metric);
  // NOTE: parsing the grid does not create the cache arrays.
  // m_super_grid.parse(m_conf_file.data().grid_config);
  // m_data_super_grid.parse(m_conf_file.data().data_grid_config);

  // setup the domain
  // setup_domain(m_args.dimx(), m_args.dimy(), m_args.dimz());

  // setup the local grid and the local data output grid
  // setup_local_grid(m_local_grid, m_super_grid, m_domain_info);
  // setup_local_grid(m_data_grid, m_data_super_grid, m_domain_info);

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
  Logger::print_info("Each particle is worth {} bytes",
                     particle_data::size);
  Logger::print_info("Each photon is worth {} bytes",
                     photon_data::size);
}
// void
// Environment::setup_domain(int num_nodes) {
//   int ndims = m_super_grid.dim();

//   //   if (total_dims > 1)
//   //     fmt::print(stderr, "Supplied rank dimensions don't match
//   total number
//   //     of ranks!");

//   // Start to create dims from scratch
//   int dims[3] = {1, 1, 1};
//   for (int i = 0; i < ndims; ++i) dims[i] = 0;

//   // m_comm->cartesian().create_dims(num_nodes, ndims, dims);

//   if (ndims == 1) setup_domain(dims[0]);
//   if (ndims == 2) setup_domain(dims[0], dims[1]);
//   if (ndims == 3) setup_domain(dims[0], dims[1], dims[2]);
// }

// void
// Environment::setup_domain(int dimx, int dimy, int dimz) {
//   // First create Cartesian rank group
//   bool periodic[3];
//   for (int i = 0; i < 3; i++) {
//     periodic[i] = m_conf_file.data().boundary_periodic[i];
//   }
//   int dims[3]{dimx, dimy, dimz};
//   auto cartesian_ranks = m_comm->get_cartesian_members(dimx * dimy *
//   dimz); m_comm->cartesian().create_cart(m_super_grid.dim(), dims,
//   periodic,
//                                   cartesian_ranks);
//   // if (!m_comm->cartesian().is_null())
//   //   m_domain_info.state = ProcessState::primary;

//   Logger::print_debug("Domain created with dimensions {} x {} x {}",
//   dimx, dimy, dimz);

//   // Initialize domain_info
//   m_domain_info.dim = m_super_grid.dim();
//   // m_domain_info.rank_map.resize(dimz);
//   m_domain_info.rank = m_comm->world().rank();
//   m_domain_info.cart_pos.x = m_comm->cartesian().coord(0);
//   // if (m_domain_info.dim >= 2)
//     // m_domain_info.cart_pos.y = m_comm->cartesian().coord(1);
//   // if (m_domain_info.dim >= 3)
//   //   m_domain_info.cart_pos.z = m_comm->cartesian().coord(2);

//   for (int i = 0; i < 3; i++) {
//     m_domain_info.cart_dims[i] = dims[i];
//   //   if (i < m_domain_info.dim) {
//   //     m_domain_info.cart_neighbor_left[i] =
//   //         m_comm->cartesian().neighbor_left(i);
//   //     // Logger::print_debug_all("On rank {}, left neighbor in dir
//   {} is ({})", m_domain_info.rank, i,
//   m_domain_info.cart_neighbor_left[i]);
//   //     if (m_domain_info.cart_neighbor_left[i] == NEIGHBOR_NULL)
//   //       m_domain_info.is_boundary[i*2] = true;

//   //     m_domain_info.cart_neighbor_right[i] =
//   //         m_comm->cartesian().neighbor_right(i);
//   //     if (m_domain_info.cart_neighbor_right[i] == NEIGHBOR_NULL)
//   //       m_domain_info.is_boundary[i*2 + 1] = true;
//   //     // Logger::print_debug_all("On rank {}, right neighbor in
//   dir {} is ({})", m_domain_info.rank, i,
//   m_domain_info.cart_neighbor_right[i]);
//   //   } else {
//   //     m_domain_info.cart_neighbor_left[i] =
//   //         m_domain_info.cart_neighbor_right[i] = NEIGHBOR_NULL;
//   //   }
//   }

//   // Debug info for boundary
//   // Logger::print_debug_all("Rank {} has boundary info {} {} {} {}",
//   m_domain_info.rank, m_domain_info.is_boundary[0],
//   m_domain_info.is_boundary[1], m_domain_info.is_boundary[2],
//   m_domain_info.is_boundary[3]);

//   // Cache whether this node is boundary on all 6 directions
//   // for (int i = 0; i < NUM_BOUNDARIES; ++i)
//   m_domain_info.is_boundary[i] = false;
//   // if (m_domain_info.cart_neighbor_left[0] == NEIGHBOR_NULL)
//   m_domain_info.is_boundary[0] = true;
//   // if (m_domain_info.cart_neighbor_right[0] == NEIGHBOR_NULL)
//   m_domain_info.is_boundary[1] = true;
//   // if (m_domain_info.cart_neighbor_left[1] == NEIGHBOR_NULL)
//   m_domain_info.is_boundary[2] = true;
//   // if (m_domain_info.cart_neighbor_right[1] == NEIGHBOR_NULL)
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
// }

// void
// Environment::set_initial_condition(InitialCondition* ic) {
//   if (m_ic == nullptr)
//     m_ic.reset(ic);
//   else
//     Logger::print_err("Initial condition already set!");
// }

// void
// Environment::set_initial_condition(SimData& data, const Index& start,
// const Extent& extent) {}

// void
// Environment::add_fieldBC(fieldBC *bc) {
//   // Only add boundary condition if the node is at boundary
//   int pos = static_cast<int>(bc->pos());
//   if (m_domain_info.is_boundary[pos]) {
//     Logger::print_debug_all("Rank {} Adding boundary condition at pos
//     {}", m_domain_info.rank, pos); m_bc.add_fieldBC(bc);
//   }
// }

// void
// Environment::add_ptcBC(ptcBC* bc) {
//   m_bc.add_ptcBC(bc);
// }

// void
// Environment::setup_local_grid(Grid &local_grid, const Grid
// &super_grid, const DomainInfo& info) {
//   local_grid = super_grid;
//   auto& local_mesh = local_grid.mesh();
//   const auto& super_mesh = super_grid.mesh();
//   if (info.cart_dims[0] > 0) {
//     local_mesh.dims[0] =
//         super_mesh.reduced_dim(0) / info.cart_dims[0] + 2 *
//         super_mesh.guard[0];
//     local_mesh.sizes[0] /= info.cart_dims[0];
//   }
//   if (info.cart_dims[1] > 0) {
//     local_mesh.dims[1] =
//         super_mesh.reduced_dim(1) / info.cart_dims[1] + 2 *
//         super_mesh.guard[1];
//     local_mesh.sizes[1] /= info.cart_dims[1];
//   }
//   if (info.cart_dims[2] > 0) {
//     local_mesh.dims[2] =
//         super_mesh.reduced_dim(2) / info.cart_dims[2] + 2 *
//         super_mesh.guard[2];
//     local_mesh.sizes[2] /= info.cart_dims[2];
//   }

//   local_mesh.lower[0] += info.cart_pos.x * local_mesh.reduced_dim(0)
//   *
//                          local_mesh.delta[0];
//   local_mesh.lower[1] += info.cart_pos.y * local_mesh.reduced_dim(1)
//   *
//                          local_mesh.delta[1];
//   local_mesh.lower[2] += info.cart_pos.z * local_mesh.reduced_dim(2)
//   *
//                          local_mesh.delta[2];
//   local_grid.gen_config();
// }

// void
// Environment::apply_initial_condition(SimData &data) {
//   Logger::print_info("Applying initial condition");
//   if (m_ic == nullptr) {
//     Logger::print_err("No initial condition set yet!");
//     return;
//   }

//   data.E.initialize([this, &data](int n, Scalar x1, Scalar x2, Scalar
//   x3){
//       return this->initial_condition().E(n, x1, x2, x3,
//       data.E.grid().mesh());
//     });

//   data.B.initialize([this, &data](int n, Scalar x1, Scalar x2, Scalar
//   x3){
//       return this->initial_condition().B(n, x1, x2, x3,
//       data.B.grid().mesh());
//     });

//   m_bc.apply_fieldBC(data.E, data.B);

//   m_ic->set_data(data);
//   m_bc.initialize(*this, data);
// }

void
Environment::check_dev_mesh(Quadmesh& mesh) {
  get_dev_mesh(mesh);
}

void
Environment::check_dev_params(SimParams& params) {
  get_dev_params(params);
}

void
Environment::init_bg_fields(SimData& data) {
  m_Ebg = data.E;
  m_Bbg = data.B;
  m_Ebg.sync_to_host();
  m_Bbg.sync_to_host();

  data.E.assign(0.0);
  data.B.assign(0.0);
  data.E.sync_to_host();
  data.B.sync_to_host();
}

}  // namespace Aperture
