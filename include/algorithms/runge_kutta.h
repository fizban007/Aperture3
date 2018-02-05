#ifndef  _RUNGEKUTTA_H_
#define  _RUNGEKUTTA_H_

#include <iostream>
#include <cmath>
#include <algorithm>
#include <array>
#include <vector>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <numeric>

#include "algorithms/interpolation.h"
#include "data/fields.h"
#include "data/quadmesh.h"
#include "data/vec3.h"
// #include "Types.h"
// #include "CoordSystem.h"

namespace Aperture {

// namespace detail {

// Here the generic Func signature should be f(x, y, dydx)
// template <typename Func, typename Data_t>
// void RungeKutta5_vec(const Vec3<Data_t>& y, const Vec3<Data_t>& dydx, const double x, const double h,
//                      Vec3<Data_t> &y_out, double &error, Vec3<Data_t> &dydx_next,
//                      // double rcont[],
//                      Func &derivs)
// {
//   //double hh = h*0.5;
//   //double xh = x + hh;
//   //double h6 = h/6.0;
//   //double yt, dyt, dym;
//   // double dyt, yout1, yout2, dym;
//   Vec3<Data_t> dyt, yout1, yout2, dym;

//   //yt = y + hh * dydx;
//   //derivs(xh, yt, dyt);
//   //yt = y + hh * dyt;
//   //derivs(xh, yt, dym);

//   //yt = y + h * dym;
//   //dym += dyt;

//   //derivs(x + h, yt, dyt);
//   //y_out = y + h6 * (dydx + dyt + 2.0*dym);
    
//   Vec3<Data_t> k1 = h * dydx;
//   derivs(x + c2 * h, y + a21 * k1, dyt);
//   Vec3<Data_t> k2 = h * dyt;
//   derivs(x + c3 * h, y + a31 * k1 + a32 * k2, dyt);
//   Vec3<Data_t> k3 = h * dyt;
//   derivs(x + c4 * h, y + a41 * k1 + a42 * k2 + a43 * k3, dyt);
//   Vec3<Data_t> k4 = h * dyt;
//   derivs(x + c5 * h, y + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, dyt);
//   Vec3<Data_t> k5 = h * dyt;
//   derivs(x + c6 * h, y + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5, dyt);
//   Vec3<Data_t> k6 = h * dyt;

//   yout1 = y + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6;
//   derivs(x + h, yout1, dym);
//   yout2 = y + bs1 * k1 + bs2 * k2 + bs3 * k3 + bs4 * k4 + bs5 * k5 + bs6 * k6 + bs7 * h * dym;

//   error = (yout2 - yout1).length();
//   dydx_next = dym;
//   y_out = yout1;

//   // rcont[0] = y;
//   // rcont[1] = yout1 - y;
//   // rcont[2] = h*dydx - rcont[1];
//   // rcont[3] = rcont[1] - h*dydx_next - rcont[2];
//   // rcont[4] = h*d1*dydx + d3*k3 + d4*k4 + d5*k5 + d6*k6 + h*d7*dydx_next;
// }

// }

template <typename Data_t, typename Type_y>
class RungeKutta
{
 public:
  RungeKutta()
      : m_dense(false), m_dense_step(0.1), m_debug_msg(false) {}
  RungeKutta(bool dense, double dense_step = 0.1, bool debug_msg = false)
      : m_dense(dense), m_dense_step(dense_step), m_debug_msg(debug_msg) {}
  ~RungeKutta() {}

  // Taking a single Runge-Kutta step and populate the results, given a function f.
  // Here the generic Func signature should be f(x, v_field, dydh). f
  // should evaluate the field at position x and return a unit vector in
  // v_field, as well as return the local rate of change in y along the
  // field direction
  template <typename Func>
  void RungeKutta5_field(const Type_y& y, const Vec3<Data_t>& v_field, const Type_y& dydh, const Vec3<Data_t>& x, double h,
                         Type_y& y_out, double& err, Vec3<Data_t>& x_next, Type_y& dydh_next, Vec3<Data_t>& v_next,
                         Func &f, std::array<Data_t, 5>& d_coef, std::array<Vec3<Data_t>, 5>& v_coef)
  {
    // Intermediate vector values and rates of change
    Vec3<Data_t> v_t, v_m;
    Type_y dydh_t, dydh_m;

    auto k1 = h * v_field;
    auto ky1 = h * dydh;
    f(x + a21 * k1, v_t, dydh_t);
    auto k2 = h * v_t;
    auto ky2 = h * dydh_t;
    f(x + a31 * k1 + a32 * k2, v_t, dydh_t);
    auto k3 = h * v_t;
    auto ky3 = h * dydh_t;
    f(x + a41 * k1 + a42 * k2 + a43 * k3, v_t, dydh_t);
    auto k4 = h * v_t;
    auto ky4 = h * dydh_t;
    f(x + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, v_t, dydh_t);
    auto k5 = h * v_t;
    auto ky5 = h * dydh_t;
    f(x + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5, v_t, dydh_t);
    auto k6 = h * v_t;
    auto ky6 = h * dydh_t;

    auto yout1 = y + b1 * ky1 + b2 * ky2 + b3 * ky3 + b4 * ky4 + b5 * ky5 + b6 * ky6;
    auto x1 = x + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6;
    f(x1, v_m, dydh_m);
    // std::cout << "V_m is " << v_m << std::endl;
    auto yout2 = y + bs1 * ky1 + bs2 * ky2 + bs3 * ky3 + bs4 * ky4 + bs5 * ky5 + bs6 * ky6 + bs7 * h * dydh_m;
    // auto x2 = x + bs1 * k1 + bs2 * k2 + bs3 * k3 + bs4 * k4 + bs5 * k5 + bs6 * k6 + bs7 * h * v_m;

    // err = (x2 - x1).length();
    err = std::abs(yout1 - yout2);
    x_next = x1;
    dydh_next = dydh_m;
    v_next = v_m;
    // y_out = h * (v_field + v_m).length() * 0.5;
    y_out = yout1;

    d_coef[0] = y;
    d_coef[1] = yout1 - y;
    d_coef[2] = ky1 - d_coef[1];
    d_coef[3] = d_coef[1] - h * dydh_next - d_coef[2];
    d_coef[4] = ky1 *d1 + d3 * ky3 + d4 * ky4 + d5 * ky5 + d6 * ky6 + h * d7 * dydh_next;

    v_coef[0] = x;
    v_coef[1] = x1 - x;
    v_coef[2] = k1 - v_coef[1];
    v_coef[3] = v_coef[1] - h * v_m - v_coef[2];
    v_coef[4] = k1 * d1 + k3 * d3 + k4 * d4 + k5 * d5 + k6 * d6 + v_m * d7 * h;
  }

  // Integrate a scalar function along a vector field, obtaining the points on
  // the field line as well as the function values on the field line
  // Func should have signature Func(Vec3<Data_t>, Vec3<Data_t>&, Type_y&)
  template <typename Func>
  Data_t integrate_along_field(const Func& f, const Quadmesh& mesh,
                               const Vec3<Data_t>& start_point,
                               double initial_step = 0.01,
                               Type_y initial_cond = 0.0) {
    std::cout << "integrating the field" << std::endl;
    // Clear the output container
    m_output_y.clear();
    m_output_points.clear();

    // Array of length along the field line
    std::vector<double> h;

    auto x_prev = start_point;
    auto y_prev = initial_cond;
    h.push_back(0.0);
    // Set the initial output point to the starting point given
    m_output_points.push_back(start_point);
    m_output_y.push_back(initial_cond);

    // This is the integration result we will return in the end
    Data_t result = 0.0;

    // Auxiliary step variables
    Vec3<Data_t> v_next;
    Data_t dydh_next;
    f(start_point, v_next, dydh_next);

    // int out_idx = 1;
    double atol = 1e-8, rtol = 1e-8;
    // double atol = 0.0, rtol = 1e-13;
    double Safe = 0.95;
    bool reject = true;
    // double h_step = 1.0 / maxSteps;
    double h_step = initial_step;
    double rel_error = 1.0;
    Vec3<float> rel_pos;
    double dense_h = m_dense_step;

    std::array<Data_t, 5> dense_coef;
    std::array<Vec3<Data_t>, 5> dense_coef_v;

    // std::cout << b1 + b2 + b3 + b4 + b5 + b6 << std::endl;

    for (int i = 0; i < maxSteps; i++) {
      double error;
      auto dydh = dydh_next;
      auto v_field = v_next;
      Data_t y_out = 0.0;
      Vec3<Data_t> x_out;

      // std::cout << "stepsize is " << h_step << std::endl;

      // See if the previous step was rejected and change the step size
      // accordingly
      if (reject == false) {
        rel_error = max(rel_error, 1.0e-8);
        // TODO: More detailed PI step size control?
        double scale = std::min(Safe * pow(rel_error, -0.2), 5.0);
        // if (scale > 5.0) scale = 5.0;
        h_step *= scale;
      } else {
        // just clear the reject flag
        reject = false;
      }

      while (true) {
        RungeKutta5_field(y_prev, v_field, dydh, x_prev, h_step, y_out, error, x_out, dydh_next, v_next, f,
                          dense_coef, dense_coef_v);
        // rel_error = error / (atol + max(output[i], y_out) * rtol);
        rel_error = error / (atol + x_out.length() * rtol);
        if (rel_error != rel_error) // Test NaN
        {
          // NaN occurred, we need to decrease step size
          h_step *= 0.2;
          continue;
        }
        // std::cout << "error is " << error << ", rel_error is " << rel_error << std::endl;
        if (m_debug_msg)
          std::cout << "At step " << i << " with result " << y_out << " x_next " << x_out << " and h_step " << h_step << " rel_error " << rel_error << " v_next " << v_next << " dydh_next " << dydh_next << std::endl;
        if (rel_error <= 1.0) {
          reject = false;
          break;
        }
        // rejected, need to decrease step size
        double scale = Safe * pow(rel_error, -0.2);
        if (scale < 0.2) scale = 0.2;
        h_step *= scale;
        // std::cout << "stepsize is " << h_step << std::endl;
        //cout << "error is " << rel_error << endl;
        if (std::abs(h_step) <= std::abs(h[i]) * eps)
          throw std::runtime_error("Stepsize underflow");
        reject = true;
      }

      // std::cout << "x_out is " << x_out << std::endl;
      // std::cout << "v_next is " << v_next << std::endl;
      // std::cout << mesh.isInBulk(mesh.findCell(x_out, rel_pos)) << std::endl;
      // std::cout << mesh.guard[0] << " " << mesh.guard[1] << " " << mesh.guard[2] << std::endl;
      // std::cout << mesh.dims[0] << " " << mesh.dims[1] << " " << mesh.dims[2] << std::endl;
      if (x_out.x < mesh.lower[0] || x_out.x > mesh.lower[0] + mesh.sizes[0] ||
          x_out.y < mesh.lower[1] || x_out.y > mesh.lower[1] + mesh.sizes[1]) break;

      // if ((mesh.findCell(x_out, rel_pos)) < 0 || mesh.findCell(x_out, rel_pos) > mesh.size()) return result;

      // std::cout << h_step << " " << y_out - output.back() << std::endl;
      if (m_debug_msg)
        std::cout << "step " << i << " final output is " << y_out << std::endl;
      // std::cout << h_step << " " << abs(x_out - x[i]) << std::endl;
      result += h_step * dydh_next;
      if (result != result) {
        std::cerr << "NaN detected, aborting!" << std::endl;
        result = 0.0;
        return 0.0;
      }
      if (y_out == y_out) {
        h.push_back(h_step);
        m_dense_coef.push_back(dense_coef);
        m_dense_coef_v.push_back(dense_coef_v);
        // Check if dense output or not
        if (m_dense) {
          // std::cout << dense_h << " " << h_step << " " << h.size() << std::endl;
          while (dense_h < h_step) {
            m_output_points.push_back(dense_out_x(dense_h, h_step));
            m_output_y.push_back(dense_out_y(dense_h, h_step));
            dense_h += m_dense_step;
          }
          if (dense_h < h_step)
            dense_h = m_dense_step - (h_step - dense_h);
          else
            dense_h = dense_h - h_step;
        } else {
          m_output_y.push_back(y_out);
          // result += sqrt(dx.x * dx.x + 0.25 * (x_out.x + x.back().x) * (x_out.x + x.back().x) * dx.y * dx.y);
          m_output_points.push_back(x_out);
        }
        y_prev = y_out;
        x_prev = x_out;
      }
      // std::cout << h.back() << std::endl;
      // std::cout << std::accumulate(h.begin(), h.end(), 0.0) << std::endl;
      // std::cout << output.back() << std::endl;
      // std::cout << result << std::endl;
    }
    // std::cout << "Result is " << result << " in " << h.size() << " steps." << std::endl;

    return result;
  }

  template <typename Func>
  Data_t integrate_along_field_simple(const Func& f, const Quadmesh& mesh,
                                      const Vec3<Data_t>& start_point,
                                      Data_t step_size = 0.01,
                                      Type_y initial_cond = 0.0) {
    std::cout << "integrating the field" << std::endl;
    // Clear the output container
    m_output_y.clear();
    m_output_points.clear();

    auto x_prev = start_point;
    auto y_prev = initial_cond;
    // Set the initial output point to the starting point given
    m_output_points.push_back(start_point);
    m_output_y.push_back(initial_cond);

    // This is the integration result we will return in the end
    Data_t result = initial_cond;

    for (int i = 0; i < maxSteps; i++) {
      Vec3<Data_t> v{};
      Data_t dydh = 0.0;
      f(x_prev, v, dydh);
      x_prev += v * step_size;
      y_prev += dydh * step_size;
      if (x_prev.x < mesh.lower[0] || x_prev.x > mesh.lower[0] + mesh.sizes[0] ||
          x_prev.y < mesh.lower[1] || x_prev.y > mesh.lower[1] + mesh.sizes[1]) break;
      m_output_points.push_back(x_prev);
      m_output_y.push_back(y_prev);
      result += y_prev;
    }
    // std::cout << "Result is " << result << " in " << m_output_y.size() << " steps." << std::endl;
    return result;
  }

  const std::vector<Vec3<Data_t> >& output_points() const { return m_output_points; }
  const std::vector<Type_y>& output_y() const { return m_output_y; }

 private:
  Vec3<Data_t> dense_out_x(double step, double h) {
    double s = step / h;
    double s1 = 1.0 - s;
    auto& coef = m_dense_coef_v.back();
    return coef[0] + s * (coef[1] + s1 * (coef[2] + s * (coef[3] + s1 * coef[4])));
  }

  Type_y dense_out_y(double step, double h) {
    double s = step / h;
    double s1 = 1.0 - s;
    auto& coef = m_dense_coef.back();
    return coef[0] + s * (coef[1] + s1 * (coef[2] + s * (coef[3] + s1 * coef[4])));
  }

  bool m_dense, m_debug_msg;
  double m_dense_step;
  std::vector<std::array<Type_y, 5> > m_dense_coef;
  std::vector<std::array<Vec3<Data_t>, 5> > m_dense_coef_v;
  std::vector<Vec3<Data_t> > m_output_points;
  std::vector<Type_y > m_output_y;

  const int maxSteps = 1000000;

  const double c2 = 1.0/5.0;
  const double c3 = 3.0/10.0;
  const double c4 = 4.0/5.0;
  const double c5 = 8.0/9.0;
  const double c6 = 1.0;

  const double d1 = -12715105075.0/11282082432.0;
  const double d3 = 87487479700.0/32700410799.0;
  const double d4 = -10690763975.0/1880347072.0;
  const double d5 = 701980252875.0/199316789632.0;
  const double d6 = -1453857185.0/822651844.0;
  const double d7 = 69997945.0/29380423.0;

  const double a21 = 1.0/5.0;

  const double a31 = 3.0/40.0;
  const double a32 = 9.0/40.0;

  const double a41 = 44.0/45.0;
  const double a42 = -56.0/15.0;
  const double a43 = 32.0/9.0;

  const double a51 = 19372.0/6561.0;
  const double a52 = -25360.0/2187.0;
  const double a53 = 64448.0/6561.0;
  const double a54 = -212.0/729.0;

  const double a61 = 9017.0/3168.0;
  const double a62 = -355.0/33.0;
  const double a63 = 46732.0/5247.0;
  const double a64 = 49.0/176.0;
  const double a65 = -5103.0/18656.0;

  const double b1 = 35.0/384.0;
  const double b2 = 0.0;
  const double b3 = 500.0/1113.0;
  const double b4 = 125.0/192.0;
  const double b5 = -2187.0/6784.0;
  const double b6 = 11.0/84.0;

  const double bs1 = 5179.0/57600.0;
  const double bs2 = 0.0;
  const double bs3 = 7571.0/16695.0;
  const double bs4 = 393.0/640.0;
  const double bs5 = -92097.0/339200.0;
  const double bs6 = 187.0/2100.0;
  const double bs7 = 1.0/40.0;

  const double eps = std::numeric_limits<double>::epsilon();

}; // ----- end of class RungeKutta -----



}

#endif   // _RUNGEKUTTA_H_
