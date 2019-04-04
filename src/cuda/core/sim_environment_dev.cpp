#include "cuda/core/sim_environment_dev.h"
#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/core/cu_sim_data.h"
#include "cuda/cudaUtility.h"
#include "cuda/grids/grid_1dgr_dev.h"
#include "cuda/grids/grid_log_sph_dev.h"
#include "cuda/utils/iterate_devices.h"
// #include "domain_communicator.h"

namespace Aperture {

// cu_sim_environment&
cu_sim_environment::cu_sim_environment(int *argc, char ***argv)
    : sim_environment(argc, argv) {
  setup_env();
}

cu_sim_environment::cu_sim_environment(const std::string &conf_file)
    : sim_environment(conf_file) {
  setup_env();
}

cu_sim_environment::~cu_sim_environment() {}

void
cu_sim_environment::setup_env() {
  sim_environment::setup_env();

  // Poll the system to detect how many GPUs are on the node
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  if (n_devices <= 0) {
    Logger::err("No usable Cuda device found!!");
    exit(1);
  }
  m_dev_map.resize(n_devices);
  Logger::print_info("Found {} Cuda devices:", n_devices);
  for (int i = 0; i < n_devices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    Logger::print_info("    Device Number: {}", i);
    Logger::print_info("    Device Name: {}", prop.name);
    Logger::print_info("    Device Total Memory: {}MiB",
                       prop.totalGlobalMem / (1024 * 1024));
    m_dev_map[i] = i;
    // if (i > 0) {
    //   int can_access_left, can_access_right;
    //   //Check for peer access between participating GPUs:
    //   cudaDeviceCanAccessPeer(&can_access_left, i, i - 1);
    //   cudaDeviceCanAccessPeer(&can_access_right, i - 1, i);
    //   if (can_access_left == 0 || can_access_right == 0) {
    //     Logger::err("Device {} cannot access its neighbor!", i);
    //   }
    // }
  }

  m_sub_params.resize(n_devices);
  m_boundary_info.resize(n_devices);
  // m_sub_buffer.resize(n_devices);
  for (int i = 0; i < n_devices; i++) {
    int dev_id = m_dev_map[i];
    CudaSafeCall(cudaSetDevice(dev_id));

    // if (i > 0)
    //   CudaSafeCall(cudaDeviceEnablePeerAccess(m_dev_map[i - 1], 0));
    // if (i < n_devices - 1)
    //   CudaSafeCall(cudaDeviceEnablePeerAccess(m_dev_map[i + 1], 0));

    m_sub_params[i] = m_params;

    int d = m_grid->dim() - 1;
    m_sub_params[i].N[d] /= n_devices;
    m_sub_params[i].size[d] /= n_devices;
    m_sub_params[i].lower[d] += i * m_sub_params[i].size[d];

    init_dev_params(m_sub_params[i]);
    // init_dev_mesh(m_grid->mesh());

    init_dev_charges(m_charges.data());
    init_dev_masses(m_masses.data());

    if (m_super_grid->dim() == 1) {
      m_sub_buffer_left.emplace_back(
          Extent{m_sub_params[i].guard[0] + 1, 1, 1});
      m_sub_buffer_right.emplace_back(
          Extent{m_sub_params[i].guard[0] + 1, 1, 1});
    } else if (m_super_grid->dim() == 2) {
      m_sub_buffer_left.emplace_back(
          Extent{m_sub_params[i].N[0] + 2 * m_sub_params[i].guard[0],
                 m_sub_params[i].guard[1] + 1, 1});
      m_sub_buffer_right.emplace_back(
          Extent{m_sub_params[i].N[0] + 2 * m_sub_params[i].guard[0],
                 m_sub_params[i].guard[1] + 1, 1});
    } else if (m_super_grid->dim() == 3) {
      m_sub_buffer_left.emplace_back(
          Extent{m_sub_params[i].N[0] + 2 * m_sub_params[i].guard[0],
                 m_sub_params[i].N[1] + 2 * m_sub_params[i].guard[1],
                 m_sub_params[i].guard[2] + 1});
      m_sub_buffer_right.emplace_back(
          Extent{m_sub_params[i].N[0] + 2 * m_sub_params[i].guard[0],
                 m_sub_params[i].N[1] + 2 * m_sub_params[i].guard[1],
                 m_sub_params[i].guard[2] + 1});
    }

    // Process boundary info
    int highest_dim = m_domain_info.dim - 1;
    for (int dim = 0; dim < highest_dim; dim++) {
      m_boundary_info[i][dim * 2] = m_domain_info.is_boundary[dim * 2];
      m_boundary_info[i][dim * 2 + 1] =
          m_domain_info.is_boundary[dim * 2 + 1];
    }
    if (i == 0) {
      m_boundary_info[i][highest_dim * 2] =
          m_domain_info.is_boundary[highest_dim * 2];
    } else if (i == n_devices - 1) {
      m_boundary_info[i][highest_dim * 2 + 1] =
          m_domain_info.is_boundary[highest_dim * 2 + 1];
    } else {
      m_boundary_info[i][highest_dim * 2] =
          m_boundary_info[i][highest_dim * 2 + 1] = false;
    }
    for (int j = 0; j < 6; j++) {
      Logger::print_debug("boundary info on dev {} at {} is {}", i, j,
                          m_boundary_info[i][j]);
    }
  }

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

void
cu_sim_environment::get_sub_guard_cells_left(cudaPitchedPtr p_src,
                                             cudaPitchedPtr p_dst,
                                             const Quadmesh &mesh_src,
                                             const Quadmesh &mesh_dst,
                                             int src_dev,
                                             int dst_dev) const {
  cudaMemcpy3DPeerParms myParms = {};
  myParms.srcPtr = p_src;
  myParms.dstPtr = p_dst;
  // myParms.kind = cudaMemcpyDefault;
  myParms.srcDevice = src_dev;
  myParms.dstDevice = dst_dev;
  if (mesh_src.dim() == 2) {
    myParms.srcPos = make_cudaPos(0, mesh_src.guard[1], 0);
    myParms.dstPos =
        make_cudaPos(0, mesh_dst.dims[1] - mesh_dst.guard[1], 0);
    myParms.extent = make_cudaExtent(mesh_src.dims[0] * sizeof(Scalar),
                                     mesh_src.guard[1], 1);
  } else if (mesh_src.dim() == 3) {
    myParms.srcPos = make_cudaPos(0, 0, mesh_src.guard[2]);
    myParms.dstPos =
        make_cudaPos(0, 0, mesh_dst.dims[2] - mesh_dst.guard[2]);
    myParms.extent =
        make_cudaExtent(mesh_src.dims[0] * sizeof(Scalar),
                        mesh_src.dims[1], mesh_src.guard[2]);
  }

  CudaSafeCall(cudaMemcpy3DPeer(&myParms));
}

void
cu_sim_environment::get_sub_guard_cells_right(cudaPitchedPtr p_src,
                                              cudaPitchedPtr p_dst,
                                              const Quadmesh &mesh_src,
                                              const Quadmesh &mesh_dst,
                                              int src_dev,
                                              int dst_dev) const {
  cudaMemcpy3DPeerParms myParms = {};
  myParms.srcPtr = p_src;
  myParms.dstPtr = p_dst;
  // myParms.kind = cudaMemcpyDefault;
  myParms.srcDevice = src_dev;
  myParms.dstDevice = dst_dev;
  if (mesh_src.dim() == 2) {
    myParms.srcPos =
        make_cudaPos(0, mesh_src.dims[1] - 2 * mesh_src.guard[1], 0);
    myParms.dstPos = make_cudaPos(0, 0, 0);
    myParms.extent = make_cudaExtent(mesh_src.dims[0] * sizeof(Scalar),
                                     mesh_src.guard[1], 1);
  } else if (mesh_src.dim() == 3) {
    myParms.srcPos =
        make_cudaPos(0, 0, mesh_src.dims[2] - 2 * mesh_src.guard[2]);
    myParms.dstPos = make_cudaPos(0, 0, 0);
    myParms.extent =
        make_cudaExtent(mesh_src.dims[0] * sizeof(Scalar),
                        mesh_src.dims[1], mesh_src.guard[2]);
  }

  CudaSafeCall(cudaMemcpy3DPeer(&myParms));
}

void
cu_sim_environment::get_sub_guard_cells(
    std::vector<cu_scalar_field<Scalar>> &field) const {
  if (m_dev_map.size() <= 1) return;
  if (m_super_grid->dim() == 1) {
    Logger::print_err("1D communication is not implemented yet!");
  } else {
    // Sending right to left
    for (unsigned int n = 1; n < m_dev_map.size(); n++) {
      CudaSafeCall(cudaSetDevice(m_dev_map[n - 1]));
      get_sub_guard_cells_left(field[n].ptr(), field[n - 1].ptr(),
                               field[n].grid().mesh(),
                               field[n - 1].grid().mesh(), n, n - 1);
    }
    // Sending left to right
    for (unsigned int n = 0; n < m_dev_map.size() - 1; n++) {
      CudaSafeCall(cudaSetDevice(m_dev_map[n + 1]));
      get_sub_guard_cells_right(field[n].ptr(), field[n + 1].ptr(),
                                field[n].grid().mesh(),
                                field[n + 1].grid().mesh(), n, n + 1);
    }
  }
}

void
cu_sim_environment::get_sub_guard_cells(
    std::vector<cu_vector_field<Scalar>> &field) const {
  if (m_dev_map.size() <= 1) return;
  if (m_super_grid->dim() == 1) {
    Logger::print_err("1D communication is not implemented yet!");
  } else {
    // Sending right to left
    for (unsigned int n = 1; n < m_dev_map.size(); n++) {
      CudaSafeCall(cudaSetDevice(m_dev_map[n - 1]));
      for (int i = 0; i < VECTOR_DIM; i++) {
        get_sub_guard_cells_left(field[n].ptr(i), field[n - 1].ptr(i),
                                 field[n].grid().mesh(),
                                 field[n - 1].grid().mesh(), n, n - 1);
      }
    }
    // Sending left to right
    for (unsigned int n = 0; n < m_dev_map.size() - 1; n++) {
      CudaSafeCall(cudaSetDevice(m_dev_map[n + 1]));
      for (int i = 0; i < VECTOR_DIM; i++) {
        get_sub_guard_cells_right(field[n].ptr(i), field[n + 1].ptr(i),
                                  field[n].grid().mesh(),
                                  field[n + 1].grid().mesh(), n, n + 1);
      }
    }
  }
}

void
cu_sim_environment::send_sub_guard_cells_left(
    cu_multi_array<Scalar> &src, cu_multi_array<Scalar> &dst,
    const Quadmesh &mesh_src, const Quadmesh &mesh_dst, int buffer_id,
    int src_dev, int dst_dev, bool stagger) const {
  cudaMemcpy3DPeerParms myParms = {};
  myParms.srcPtr = src.data_d();
  myParms.dstPtr = m_sub_buffer_left[dst_dev].data_d();
  // myParms.kind = cudaMemcpyDefault;
  myParms.srcDevice = src_dev;
  myParms.dstDevice = dst_dev;
  if (mesh_src.dim() == 2) {
    myParms.srcPos = make_cudaPos(0, 0, 0);
    myParms.dstPos = make_cudaPos(0, 0, 0);
    myParms.extent = make_cudaExtent(mesh_src.dims[0] * sizeof(Scalar),
                                     mesh_src.guard[1], 1);
  } else if (mesh_src.dim() == 3) {
    myParms.srcPos = make_cudaPos(0, 0, 0);
    myParms.dstPos = make_cudaPos(0, 0, 0);
    myParms.extent =
        make_cudaExtent(mesh_src.dims[0] * sizeof(Scalar),
                        mesh_src.dims[1], mesh_src.guard[2]);
  }

  CudaSafeCall(cudaMemcpy3DPeer(&myParms));
  // Logger::print_debug("Finished copying to peer");
  // Add the buffer to the target
  // if (mesh_src.dim() == 2) {
  //   // CudaSafeCall(cudaSetDevice(m_dev_map[buffer_id]));
  //   dst.add_from(m_sub_buffer_left[buffer_id], Index{0, 0, 0},
  //                Index{0, mesh_dst.dims[1] - 2 * mesh_dst.guard[1],
  //                0}, Extent{mesh_dst.dims[0], mesh_dst.guard[1], 1});
  // } else if (mesh_src.dim() == 3) {
  //   // CudaSafeCall(cudaSetDevice(m_dev_map[buffer_id]));
  //   dst.add_from(
  //       m_sub_buffer_left[buffer_id], Index{0, 0, 0},
  //       Index{0, 0, mesh_dst.dims[2] - 2 * mesh_dst.guard[2]},
  //       Extent{mesh_dst.dims[0], mesh_dst.dims[1],
  //       mesh_dst.guard[1]});
  // }
  // Logger::print_debug("Finished add_from");
}

void
cu_sim_environment::send_sub_guard_cells_right(
    cu_multi_array<Scalar> &src, cu_multi_array<Scalar> &dst,
    const Quadmesh &mesh_src, const Quadmesh &mesh_dst, int buffer_id,
    int src_dev, int dst_dev, bool stagger) const {
  cudaMemcpy3DPeerParms myParms = {};
  myParms.srcPtr = src.data_d();
  myParms.dstPtr = m_sub_buffer_right[dst_dev].data_d();
  // myParms.kind = cudaMemcpyDefault;
  myParms.srcDevice = src_dev;
  myParms.dstDevice = dst_dev;
  if (mesh_src.dim() == 2) {
    myParms.srcPos = make_cudaPos(
        0, mesh_src.dims[1] - mesh_src.guard[1] - (stagger ? 1 : 0), 0);
    myParms.dstPos = make_cudaPos(0, 0, 0);
    myParms.extent =
        make_cudaExtent(mesh_src.dims[0] * sizeof(Scalar),
                        mesh_src.guard[1] + (stagger ? 1 : 0), 1);
  } else if (mesh_src.dim() == 3) {
    myParms.srcPos = make_cudaPos(
        0, 0, mesh_src.dims[2] - mesh_src.guard[2] - (stagger ? 1 : 0));
    myParms.dstPos = make_cudaPos(0, 0, 0);
    myParms.extent = make_cudaExtent(
        mesh_src.dims[0] * sizeof(Scalar), mesh_src.dims[1],
        mesh_src.guard[2] + (stagger ? 1 : 0));
  }

  CudaSafeCall(cudaMemcpy3DPeer(&myParms));
}

void
cu_sim_environment::add_from_buffer_left(cu_multi_array<Scalar> &dst,
                                         const Quadmesh &mesh_dst,
                                         int buffer_id,
                                         bool stagger) const {
  // Add the buffer to the target
  if (mesh_dst.dim() == 2) {
    // Logger::print_debug("add left, buffer id is {}", buffer_id);
    // CudaSafeCall(cudaSetDevice(m_dev_map[buffer_id]));
    dst.add_from(m_sub_buffer_left[buffer_id], Index{0, 0, 0},
                 Index{0, mesh_dst.dims[1] - 2 * mesh_dst.guard[1], 0},
                 Extent{mesh_dst.dims[0], mesh_dst.guard[1], 1});
  } else if (mesh_dst.dim() == 3) {
    // CudaSafeCall(cudaSetDevice(m_dev_map[buffer_id]));
    dst.add_from(
        m_sub_buffer_left[buffer_id], Index{0, 0, 0},
        Index{0, 0, mesh_dst.dims[2] - 2 * mesh_dst.guard[2]},
        Extent{mesh_dst.dims[0], mesh_dst.dims[1], mesh_dst.guard[2]});
  }
}

void
cu_sim_environment::add_from_buffer_right(cu_multi_array<Scalar> &dst,
                                          const Quadmesh &mesh_dst,
                                          int buffer_id,
                                          bool stagger) const {
  // Add the buffer to the target
  if (mesh_dst.dim() == 2) {
    // CudaSafeCall(cudaSetDevice(m_dev_map[buffer_id]));
    // Logger::print_debug("add right, buffer id is {}, extent is
    // {}x{}x1",
    //                     buffer_id, mesh_dst.dims[0],
    //                     mesh_dst.guard[1] + (stagger ? 1 : 0));
    dst.add_from(m_sub_buffer_right[buffer_id], Index{0, 0, 0},
                 Index{0, mesh_dst.guard[1] - (stagger ? 1 : 0), 0},
                 Extent{mesh_dst.dims[0],
                        mesh_dst.guard[1] + (stagger ? 1 : 0), 1});
  } else if (mesh_dst.dim() == 3) {
    // CudaSafeCall(cudaSetDevice(m_dev_map[buffer_id]));
    dst.add_from(m_sub_buffer_right[buffer_id], Index{0, 0, 0},
                 Index{0, 0, mesh_dst.guard[2] - (stagger ? 1 : 0)},
                 Extent{mesh_dst.dims[0], mesh_dst.dims[1],
                        mesh_dst.guard[2] + (stagger ? 1 : 0)});
  }
}

void
cu_sim_environment::send_sub_guard_cells(
    std::vector<cu_scalar_field<Scalar>> &field) const {
  if (m_dev_map.size() <= 1) return;
  if (m_super_grid->dim() == 1) {
    Logger::print_err("1D communication is not implemented yet!");
  } else {
    int last_dim = m_super_grid->dim() - 1;
    // Sending right to left
    for (unsigned int n = 1; n < m_dev_map.size(); n++) {
      CudaSafeCall(cudaSetDevice(m_dev_map[n - 1]));
      send_sub_guard_cells_left(
          field[n].data(), field[n - 1].data(), field[n].grid().mesh(),
          field[n - 1].grid().mesh(), n - 1, m_dev_map[n],
          m_dev_map[n - 1], field[n].stagger()[last_dim]);
    }
    // Sending left to right
    for (unsigned int n = 0; n < m_dev_map.size() - 1; n++) {
      CudaSafeCall(cudaSetDevice(m_dev_map[n + 1]));
      send_sub_guard_cells_right(
          field[n].data(), field[n + 1].data(), field[n].grid().mesh(),
          field[n + 1].grid().mesh(), n + 1, m_dev_map[n],
          m_dev_map[n + 1], field[n].stagger()[last_dim]);
    }
    for_each_device(m_dev_map, [](int n) {
      CudaSafeCall(cudaDeviceSynchronize());
    });
    for (unsigned int n = 1; n < m_dev_map.size(); n++) {
      CudaSafeCall(cudaSetDevice(m_dev_map[n - 1]));
      add_from_buffer_left(field[n - 1].data(),
                           field[n - 1].grid().mesh(), n - 1,
                           field[n - 1].stagger()[last_dim]);
    }
    for (unsigned int n = 0; n < m_dev_map.size() - 1; n++) {
      CudaSafeCall(cudaSetDevice(m_dev_map[n + 1]));
      add_from_buffer_right(field[n + 1].data(),
                            field[n + 1].grid().mesh(), n + 1,
                            field[n + 1].stagger()[last_dim]);
    }
    for_each_device(m_dev_map, [](int n) {
      CudaSafeCall(cudaDeviceSynchronize());
    });
  }
}

void
cu_sim_environment::send_sub_guard_cells(
    std::vector<cu_vector_field<Scalar>> &field) const {
  if (m_dev_map.size() <= 1) return;
  if (m_super_grid->dim() == 1) {
    Logger::print_err("1D communication is not implemented yet!");
  } else {
    int last_dim = m_super_grid->dim() - 1;
    // Sending right to left
    for (int i = 0; i < VECTOR_DIM; i++) {
      for (unsigned int n = 1; n < m_dev_map.size(); n++) {
        CudaSafeCall(cudaSetDevice(m_dev_map[n - 1]));
        send_sub_guard_cells_left(
            field[n].data(i), field[n - 1].data(i),
            field[n].grid().mesh(), field[n - 1].grid().mesh(), n - 1,
            n, n - 1, field[n].stagger(i)[last_dim]);
      }
      // Sending left to right
      for (unsigned int n = 0; n < m_dev_map.size() - 1; n++) {
        CudaSafeCall(cudaSetDevice(m_dev_map[n + 1]));
        send_sub_guard_cells_right(
            field[n].data(i), field[n + 1].data(i),
            field[n].grid().mesh(), field[n + 1].grid().mesh(), n + 1,
            n, n + 1, field[n].stagger(i)[last_dim]);
      }
      for_each_device(m_dev_map,
                      [](int n) { cudaDeviceSynchronize(); });
      for (unsigned int n = 1; n < m_dev_map.size(); n++) {
        CudaSafeCall(cudaSetDevice(m_dev_map[n - 1]));
        add_from_buffer_left(field[n - 1].data(i),
                             field[n - 1].grid().mesh(), n - 1,
                             field[n - 1].stagger(i)[last_dim]);
      }
      for (unsigned int n = 0; n < m_dev_map.size() - 1; n++) {
        CudaSafeCall(cudaSetDevice(m_dev_map[n + 1]));
        add_from_buffer_right(field[n + 1].data(i),
                              field[n + 1].grid().mesh(), n + 1,
                              field[n + 1].stagger(i)[last_dim]);
      }
      for_each_device(m_dev_map,
                      [](int n) { cudaDeviceSynchronize(); });
    }
  }
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
// cu_sim_environment::set_initial_condition(cu_sim_data& data, const
// Index& start, const Extent& extent) {}

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
cu_sim_environment::check_dev_mesh(Quadmesh &mesh) {
  get_dev_mesh(mesh);
}

void
cu_sim_environment::check_dev_params(SimParams &params) {
  get_dev_params(params);
}

// void
// cu_sim_environment::init_bg_fields(cu_sim_data& data) {
//   // Initialize the background fields
//   if (m_params.use_bg_fields) {
//     data.init_bg_fields();
//   }
// }

}  // namespace Aperture
