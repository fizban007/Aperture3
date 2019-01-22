#include "sim_environment_dev.h"
#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cudaUtility.h"
#include "data/grid_1dGR.h"
#include "data/grid_log_sph_dev.h"
// #include "data/detail/grid_impl.hpp"
#include "cu_sim_data.h"
// #include "domain_communicator.h"

namespace Aperture {

// cu_sim_environment&
cu_sim_environment::cu_sim_environment(int* argc, char*** argv)
    : sim_environment(argc, argv) {
  setup_env();
}

cu_sim_environment::cu_sim_environment(const std::string& conf_file)
    : sim_environment(conf_file) {
  setup_env();
}

cu_sim_environment::~cu_sim_environment() {}

void
cu_sim_environment::setup_env() {
  sim_environment::setup_env();

  if (m_params.coord_system == "LogSpherical") {
    m_grid.reset(new Grid_LogSph_dev());
    m_grid->init(m_params);
  }

  init_dev_params(m_params);
  init_dev_mesh(*(m_grid->mesh_ptr()));

  init_dev_charges(m_charges.data());
  init_dev_masses(m_masses.data());

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
  // Logger::print_debug("Current rank is {}", m_comm->world().rank());
}
// void
// cu_sim_environment::setup_domain(int num_nodes) {
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
// cu_sim_environment::setup_domain(int dimx, int dimy, int dimz) {
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
// cu_sim_environment::set_initial_condition(InitialCondition* ic) {
//   if (m_ic == nullptr)
//     m_ic.reset(ic);
//   else
//     Logger::print_err("Initial condition already set!");
// }

// void
// cu_sim_environment::set_initial_condition(cu_sim_data& data, const Index& start,
// const Extent& extent) {}

// void
// cu_sim_environment::add_fieldBC(fieldBC *bc) {
//   // Only add boundary condition if the node is at boundary
//   int pos = static_cast<int>(bc->pos());
//   if (m_domain_info.is_boundary[pos]) {
//     Logger::print_debug_all("Rank {} Adding boundary condition at pos
//     {}", m_domain_info.rank, pos); m_bc.add_fieldBC(bc);
//   }
// }

// void
// cu_sim_environment::add_ptcBC(ptcBC* bc) {
//   m_bc.add_ptcBC(bc);
// }

// void
// cu_sim_environment::setup_local_grid(Grid &local_grid, const Grid
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
// cu_sim_environment::apply_initial_condition(cu_sim_data &data) {
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
cu_sim_environment::check_dev_mesh(Quadmesh& mesh) {
  get_dev_mesh(mesh);
}

void
cu_sim_environment::check_dev_params(SimParams& params) {
  get_dev_params(params);
}

void
cu_sim_environment::init_bg_fields(cu_sim_data& data) {
  // Initialize the background fields
  if (m_params.use_bg_fields) {
    // data.Ebg = cu_vector_field<Scalar>(*m_grid);
    // data.Bbg = cu_vector_field<Scalar>(*m_grid);
    init_dev_bg_fields(data.Ebg, data.Bbg);

    data.Ebg = data.E;
    data.Bbg = data.B;
    data.Ebg.sync_to_host();
    data.Bbg.sync_to_host();

    data.E.assign(0.0);
    data.B.assign(0.0);
    data.E.sync_to_host();
    data.B.sync_to_host();
  }
}

}  // namespace Aperture
