#include "radiation/inverse_compton.h"
#include "data/typedefs.h"
#include "utils/util_functions.h"
#include "cuda/cuda_control.h"
#include "cuda/cudaUtility.h"
#include "cuda/constant_mem.h"
#include "sim_environment.h"
#include <curand_kernel.h>

namespace Aperture {

namespace Kernels {

__device__ __forceinline__
Scalar compute_A1(Scalar er) {
  Scalar alpha = dev_params.spectral_alpha;
  Scalar A1 = 1.0 / (er * (0.5 + 1.0 / alpha - (1.0 / (alpha * (alpha + 1.0)))
                    * std::pow(er / dev_params.e_s, alpha)));
  return A1;
}

__device__ __forceinline__
Scalar compute_A2(Scalar er, Scalar et) {
  Scalar alpha = dev_params.spectral_alpha;
  Scalar A2 = 1.0 / (et * (et * 0.5 / er + std::log(er / et) + 1.0 / (1.0 + alpha)));
  return A2;
}

__device__
Scalar f_inv1(float u, Scalar gamma) {
  Scalar alpha = dev_params.spectral_alpha;
  Scalar e_s = dev_params.e_s;
  Scalar er = 2.0 * gamma * dev_params.e_min;
  Scalar A1 = compute_A1(er);
  if (u < A1 * er * 0.5)
    return std::sqrt(2.0 * u * er / A1);
  else if (u < 1.0 - A1 * er * std::pow(e_s / er, -alpha) / (1.0 + alpha))
    return er * std::pow(alpha * (1.0 / alpha + 0.5 - u / (A1 * er)), -1.0 / alpha);
  else
    return er * std::pow((1.0 - u)*(1.0 + alpha) / (A1 * e_s), -1.0 / (alpha + 1.0));
}

__device__
Scalar f_inv2(float u, Scalar gamma) {
  Scalar alpha = dev_params.spectral_alpha;
  // Scalar e_s = dev_params.e_s;
  Scalar er = 2.0 * gamma * dev_params.e_min;
  Scalar et = er / (2.0 * er + 1.0);
  Scalar A2 = compute_A2(er, et);
  if (u < A2 * et * et * 0.5 / er)
    return std::sqrt(2.0 * u * er / A2);
  else if (u < 1.0 - A2 * et / (1.0 + alpha))
    return et * std::exp(u / (A2 * et) - et * 0.5 / er);
  else
    return er * std::pow((1.0 - u)*(1.0 + alpha) / (A2 * et), -1.0 / (alpha + 1.0));
}

// Draw the rest frame photon energy
__device__
Scalar draw_photon_e1p(Scalar gamma, curandState& state) {
  float u = curand_uniform(&state);
  Scalar e1p;
  if (gamma < dev_params.e_s * 0.5 / dev_params.e_min) {
    e1p = f_inv1(u, gamma);
  } else {
    e1p = f_inv2(u, gamma);
  }
  return e1p;
}

// Given rest frame photon energy, draw its original energy
__device__
Scalar draw_photon_ep(Scalar e1p, Scalar gamma, curandState& state) {
  double ep;
  float u = curand_uniform(&state);
  double gemin2 = 2.0 * gamma * dev_params.e_min;
  Scalar alpha = dev_params.spectral_alpha;
  if (e1p < 0.5 && e1p / (1.0 - 2.0 * e1p) <= gemin2) {
    double e_lim = e1p / (1.0 - 2.0 * e1p);
    double a1 = (gemin2 * gemin2 * (alpha + 2.0)) / (gamma * (e_lim*e_lim - e1p*e1p));
    ep = std::sqrt(u * (alpha + 2.0) * gemin2 * gemin2 / (a1 * gamma) + e1p*e1p);
  } else if (e1p > gemin2) {
    double a2 = (alpha * (alpha + 2.0) * 0.5 / gamma) * std::pow(e1p / gemin2, alpha);
    if (e1p < 0.5)
      a2 /= (1.0 - std::pow(1.0 - 2.0 * e1p, alpha));
    ep = gemin2 * std::pow(std::pow(gemin2/e1p, alpha) - u * alpha * (alpha + 2.0) / (2.0 * gamma * a2), -1.0/alpha);
  } else {
    double G = 0.0;
    if (e1p < 0.5)
      G = std::pow((1.0 - 2.0 * e1p) * gemin2 / e1p, alpha);
    double U_0 = (gemin2*gemin2 - e1p*e1p)*gamma/(gemin2*gemin2*(alpha + 2.0));
    double a3 = 1.0 / (U_0 + (1.0 - G)*2.0*gamma/(alpha * (alpha + 2.0)));
    if (u < U_0 * a3)
      ep = std::sqrt(u * (alpha + 2.0) * gemin2 * gemin2 / (a3 * gamma) + e1p*e1p);
    else
      ep = gemin2 * std::pow(1.0 - (u - a3 * U_0) * alpha * (alpha + 2.0) / (2.0 * a3 * gamma), -1.0/alpha);
  }
  return ep;
}

// Given rest frame photon energy, draw the rest frame photon angle
__device__
Scalar draw_photon_u1p(Scalar e1p, Scalar gamma, curandState& state) {
  Scalar u1p;
  Scalar ep = draw_photon_ep(e1p, gamma, state);
  u1p = 1.0 - 1.0 / e1p + 1.0 / ep;
  return u1p;
}

// Draw the lab frame photon energy
__device__
Scalar draw_photon_energy(Scalar gamma, Scalar p, Scalar x, curandState& state) {
  Scalar e1p = draw_photon_e1p(gamma, state);
  Scalar u1p = draw_photon_u1p(e1p, gamma, state);

  return sgn(p) * (gamma + std::abs(p) * (-u1p)) * e1p;
}

__global__
void init_rand_states(curandState* states, int seed) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, id, 0, &states[id]);
}


}

InverseCompton::InverseCompton(const Environment& env) :
    m_env(env), m_I0(env.local_grid()) {
  const int seed = 4321;
  m_threadsPerBlock = 512;
  m_blocksPerGrid = 512;

  CudaSafeCall(cudaMalloc(&d_rand_states, m_threadsPerBlock * m_blocksPerGrid *
                          sizeof(curandState)));
  Kernels::init_rand_states<<<m_blocksPerGrid, m_threadsPerBlock>>>
      ((curandState*)d_rand_states, seed);
  CudaCheckError();
}

InverseCompton::~InverseCompton() {
  cudaFree(d_rand_states);
}

void
InverseCompton::convert_pairs(Particles& particles, Photons& photons) {

}

void
InverseCompton::emit_photons(Photons& photons, Particles& particles) {

}

void
InverseCompton::set_I0(const ScalarField<Scalar>& I0) {
  m_I0 = I0;
  m_I0.sync_to_device();
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
}

}
