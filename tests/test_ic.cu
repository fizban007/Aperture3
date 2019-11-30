#include "cuda/radiation/rt_ic.h"
#include "cuda/radiation/rt_ic_dev.h"
#include "sim_params.h"
#include "core/array.h"
#include "cuda/cudarng.h"
#include "cuda/cudaUtility.h"
#include "utils/logger.h"
#include "cuda/kernels.h"
#include "radiation/spectra.h"
#include "catch.hpp"

using namespace Aperture;

__global__ void
knl_init_ptc(Scalar* ptc_E) {}

__global__ void
knl_ic_scatter(Scalar* ph_E, const Scalar* ptc_E, curandState* states, size_t N) {
  int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N)
    ph_E[n] = Kernels::gen_photon_e(ptc_E[n], &states[n]);
}

TEST_CASE("testing ic scatter", "[IC]") {
  SimParams params;
  params.e_min = 1.0e-5;
  params.ic_path = 0.01;

  inverse_compton rt_ic(params);
  Logger::print_debug("in rt_init, emin is {}", params.e_min);
  Spectra::broken_power_law rt_ne(1.25, 1.1, params.e_min,
                                         1.0e-10, 0.1);
  rt_ic.init(rt_ne, rt_ne.emin(), rt_ne.emax(),
             1.50e24 / params.ic_path);

  const int n_threads = 256;
  const int n_blocks = 256;
  curandState* states;
  CudaSafeCall(cudaMalloc(&states, n_threads * n_blocks * sizeof(curandState)));
  init_rand_states(states, 1234, n_blocks, n_threads);

  size_t N = 100;
  array<Scalar> ptc_E(N);
  array<Scalar> ph_E(N);
  ptc_E.assign(0.0f);
  ph_E.assign(0.0f);

  for (size_t i = 0; i < N; i++) {
    ptc_E[i] = 10.0;
  }
  ptc_E.copy_to_device();

  knl_ic_scatter<<<n_blocks, n_threads>>>(ph_E.dev_ptr(), ptc_E.dev_ptr(), states, N);
  CudaCheckError();

  ph_E.copy_to_host();

  auto &log_ep = rt_ic.log_ep();
  // for (int i = 0; i < log_ep.size(); i++) {
  //   Logger::print_info("log_ep[{}] is {}, dNde is {}", i, log_ep[i], rt_ic.dNde_thomson()(i, 40));
  // }
  // for (int i = 0; i < rt_ic.gammas().size(); i++) {
  //   Logger::print_info("gammas[{}] is {}", i, rt_ic.gammas()[i]);
  // }

  for (size_t i = 0; i < N; i++) {
    Logger::print_info("Eph is {}", ph_E[i]);
  }

  CudaSafeCall(cudaFree(states));
}
