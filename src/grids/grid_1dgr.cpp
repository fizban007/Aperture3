#include "grid_1dgr.h"
#include "utils/logger.h"
#include "utils/util_functions.h"
#define H5_USE_BOOST
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <vector>

namespace HF = HighFive;

namespace Aperture {

Grid_1dGR::Grid_1dGR() : Grid() {}

Grid_1dGR::Grid_1dGR(int N) : Grid(N, 1, 1) {}

Grid_1dGR::~Grid_1dGR() {}

void
Grid_1dGR::compute_coef(const SimParams& params) {
  Logger::print_info("Resizing arrays");
  uint32_t n_size = m_mesh.dims[0];
  m_D1.resize(n_size);
  m_D2.resize(n_size);
  m_D3.resize(n_size);
  m_alpha.resize(n_size);
  m_theta.resize(n_size);
  m_K1.resize(n_size);
  m_K1_j.resize(n_size);
  m_j0.resize(n_size);
  m_agrr.resize(n_size);
  m_agrf.resize(n_size);
  m_rho0.resize(n_size);
  m_gamma_rr.resize(n_size);
  m_gamma_ff.resize(n_size);
  m_beta_phi.resize(n_size);
  m_B3B1.resize(n_size);

  Logger::print_info("Reading h5");
  HF::File coef_file("coef.h5", HF::File::ReadOnly);
  HF::DataSet data_B1 = coef_file.getDataSet("Bu1");
  uint32_t size = data_B1.getSpace().getDimensions()[0];
  // Logger::print_info("Storage size is {}",
  // data_B1.getSpace().getDimensions()[0]);
  HF::DataSet data_B3 = coef_file.getDataSet("Bu3");
  HF::DataSet data_jr = coef_file.getDataSet("ju1");
  HF::DataSet data_rho = coef_file.getDataSet("rho");
  HF::DataSet data_r = coef_file.getDataSet("r");
  HF::DataSet data_theta = coef_file.getDataSet("th");
  HF::DataSet data_dpsidth = coef_file.getDataSet("dpsidth");

  std::vector<float> v_B1(size);
  std::vector<float> v_B3(size);
  std::vector<float> v_jr(size);
  std::vector<float> v_rho(size);
  std::vector<float> v_r(size);
  std::vector<float> v_theta(size);
  std::vector<float> v_dpsidth(size);
  data_B1.read(v_B1.data());
  data_B3.read(v_B3.data());
  data_jr.read(v_jr.data());
  data_rho.read(v_rho.data());
  data_r.read(v_r.data());
  data_theta.read(v_theta.data());
  data_dpsidth.read(v_dpsidth.data());

  Scalar omega, a;
  HF::DataSet data_omega = coef_file.getDataSet("w");
  data_omega.read(omega);
  HF::DataSet data_a = coef_file.getDataSet("a");
  data_a.read(a);
  Logger::print_info("------- omega is {}, a is {}", omega, a);

  // TODO: extrapolate the coefficients to the device arrays
  uint32_t n_data = 0;
  uint32_t n_mid = 0;
  for (uint32_t n = 0; n < n_size; n++) {
    double xi = m_mesh.lower[0] +
                ((int)n - m_mesh.guard[0] + 1) * m_mesh.delta[0];
    double rp = 1.0 + std::sqrt(1.0 - a * a);
    double rm = 1.0 - std::sqrt(1.0 - a * a);
    double exp_xi = std::exp(xi * (rp - rm));
    double r = (rp - rm * exp_xi) / (1.0 - exp_xi);
    while (v_r[n_data + 1] < r) n_data += 1;
    // Logger::print_info("r is {}, vr is {}, n_data is {}, n is {}", r,
    // v_r[n_data], n_data, n);
    double x = (r - v_r[n_data]) / (v_r[n_data + 1] - v_r[n_data]);
    Scalar B1 = v_B1[n_data] * (1.0 - x) + v_B1[n_data + 1] * x;
    Scalar B3 = v_B3[n_data] * (1.0 - x) + v_B3[n_data + 1] * x;
    Scalar theta =
        v_theta[n_data] * (1.0 - x) + v_theta[n_data + 1] * x;
    Scalar b31 = B3 / B1;
    m_theta[n] = theta;
    m_B3B1[n] = b31;
    m_rho0[n] =
        params.B0 * (v_rho[n_data] * (1.0 - x) + v_rho[n_data + 1] * x);

    // Logger::print_debug("xi is {}, r is {}, x is {}, theta is {}",
    // xi, r, x, theta);
    Scalar Sigma = r * r + a * a * square(std::cos(theta));
    Scalar Delta = r * r - 2.0 * r + a * a;
    Scalar A =
        square(r * r + a * a) - Delta * a * a * square(std::sin(theta));
    Scalar w = 2.0 * a * r / A;
    Scalar g11 = Sigma / Delta;
    Scalar g22 = Sigma;
    Scalar g33 = A * square(std::sin(theta)) / Sigma;
    Scalar sqrt_gamma = std::sqrt(g11 * g22 * g33);
    // Logger::print_debug("Sigma {}, Delta {}, A {}, sqrt_gamma {}",
    // Sigma, Delta, A, sqrt_gamma);

    m_K1[n] =
        Delta * sqrt_gamma /
        (v_dpsidth[n_data] * (1.0 - x) + v_dpsidth[n_data + 1] * x);
    // (v_dpsidth[n_data] * (1.0 - x) + v_dpsidth[n_data + 1] * x) /
    // sqrt_gamma;

    // m_D1[n] = (c1 * c2 / g11 + c3 * c4 / g33) * Delta;
    m_D1[n] = (b31 * g33 * (omega - w)) * Delta;
    // m_D2[n] = (c2 * c2 / g11 + c3 * c3 / g33) * Delta * Delta;
    m_D2[n] = (g11 + b31 * b31 * g33) * Delta * Delta;
    // m_D3[n] = c1 * c1 / g11 + c4 * c4 / g33;
    m_D3[n] = g33 * square(omega - w);
    m_alpha[n] = std::sqrt(Delta * Sigma / A);
    // m_agrr[n] = std::sqrt(m_alpha[n]) * g11;
    m_gamma_rr[n] = 1.0 / g11;
    m_gamma_ff[n] = 1.0 / g33;
    m_beta_phi[n] = -2.0 * a * r / A;

    double xi_mid = m_mesh.lower[0] +
                    ((int)n - m_mesh.guard[0] + 0.5) * m_mesh.delta[0];
    double exp_xi_mid = std::exp(xi_mid * (rp - rm));
    double r_mid = (rp - rm * exp_xi_mid) / (1.0 - exp_xi_mid);
    while (v_r[n_mid + 1] < r_mid) n_mid += 1;
    double y = (r - v_r[n_mid]) / (v_r[n_mid + 1] - v_r[n_mid]);
    m_j0[n] = params.B0 *
              (v_jr[n_mid] * (1.0 - y) + v_jr[n_mid + 1] * y) / Delta;
    Scalar theta_mid =
        v_theta[n_mid] * (1.0 - y) + v_theta[n_mid + 1] * y;
    Sigma = r_mid * r_mid + a * a * square(std::cos(theta_mid));
    Delta = r_mid * r_mid - 2.0 * r_mid + a * a;
    A = square(r_mid * r_mid + a * a) -
        Delta * a * a * square(std::sin(theta_mid));
    g11 = Sigma / Delta;
    g22 = Sigma;
    g33 = A * square(std::sin(theta_mid)) / Sigma;
    sqrt_gamma = std::sqrt(g11 * g22 * g33);

    m_K1_j[n] =
        Delta * sqrt_gamma /
        (v_dpsidth[n_mid] * (1.0 - y) + v_dpsidth[n_mid + 1] * y);
    // (v_dpsidth[n_mid] * (1.0 - y) + v_dpsidth[n_mid + 1] * y) /
    // sqrt_gamma;
    m_agrr[n] = Delta * Delta * std::sqrt(Delta * Sigma / A) * g11;
    // Scalar denom = (m_alpha2[n] - m_D3[n]) * m_D2[n] +
    // square(m_D1[n]);
    // if (n % 10 == 0) {
    //   Logger::print_info(
    //       "rho0 is {}, rho0*K1 is {}, sqrt_gamma is {}, j0 is "
    //       "{}, K1 is "
    //       "{}",
    //       m_rho0[n], m_rho0[n] * m_K1[n], sqrt_gamma, m_j0[n], m_K1[n]);
    // }
  }
  m_D1.sync_to_device();
  m_D2.sync_to_device();
  m_D3.sync_to_device();
  m_alpha.sync_to_device();
  m_theta.sync_to_device();
  m_K1.sync_to_device();
  m_K1_j.sync_to_device();
  m_j0.sync_to_device();
  m_rho0.sync_to_device();
  m_agrr.sync_to_device();
  m_agrf.sync_to_device();
  m_gamma_rr.sync_to_device();
  m_gamma_ff.sync_to_device();
  m_beta_phi.sync_to_device();
  m_B3B1.sync_to_device();

  // HF::File test_out("debug.h5", HF::File::ReadWrite | HF::File::Create |
  //                                   HF::File::Truncate);
  // HF::DataSet out_K1 = test_out.createDataSet<Scalar>(
  //     "dpsidth", HF::DataSpace{m_K1_j.size()});
  // out_K1.write(m_K1_j.data());
}

Grid_1dGR::mesh_ptrs
Grid_1dGR::get_mesh_ptrs() const {
  mesh_ptrs ptrs;

  ptrs.D1 = m_D1.dev_ptr();
  ptrs.D2 = m_D2.dev_ptr();
  ptrs.D3 = m_D3.dev_ptr();
  ptrs.alpha = m_alpha.dev_ptr();
  ptrs.theta = m_theta.dev_ptr();
  ptrs.K1 = m_K1.dev_ptr();
  ptrs.K1_j = m_K1_j.dev_ptr();
  ptrs.j0 = m_j0.dev_ptr();
  ptrs.agrr = m_agrr.dev_ptr();
  ptrs.agrf = m_agrf.dev_ptr();
  ptrs.rho0 = m_rho0.dev_ptr();
  ptrs.gamma_rr = m_gamma_rr.dev_ptr();
  ptrs.gamma_ff = m_gamma_ff.dev_ptr();
  ptrs.beta_phi = m_beta_phi.dev_ptr();
  ptrs.B3B1 = m_B3B1.dev_ptr();

  return ptrs;
}

}  // namespace Aperture
