#ifndef _INTEGRATOR_IMPL_H_
#define _INTEGRATOR_IMPL_H_

#include "utils/integrator.h"
#include <iostream>

namespace Aperture {

template <typename Metric>
void
field_func<Metric>::operator()(const Vec3<double>& pos, Vec3<double>& v,
                               double& dydh) const {
  Vec3<float> rel_pos;
  auto grid = E.grid();
  // std::cout << pos << std::endl;
  int cell = grid.mesh().find_cell(pos, rel_pos);
  Vec3<int> c = grid.mesh().get_cell_3d(cell);
  // std::cout << "cell id " << cell << " is " << c << std::endl;
  // Vec3<float> rel_pos_dual = rel_pos;
  // grid.mesh().pos_dual(c_dual, rel_pos_dual);
  auto v_E = E.interpolate(c, rel_pos, 1);
  auto v_B = B.interpolate(c, rel_pos, 1);
  v = v_B;
  v.normalize();
  // std::cout << v_E << " " << v_B << " " << v << std::endl;
  v.x /= std::sqrt(g.g11(pos.x, pos.y, pos.z));
  v.y /= std::sqrt(g.g22(pos.x, pos.y, pos.z));
  v.z /= std::sqrt(g.g33(pos.x, pos.y, pos.z));

  if (component == "EdotB") {
    dydh = v_E.dot(v_B) / v_B.length();
  } else if (component == "Twist") {
    double phi = v_B.z / v_B.length();
    dydh = phi / std::sqrt(g.g33(pos.x, pos.y, pos.z));
  } else if (component == "dIdf") {
    auto v_J = J.interpolate(c, rel_pos, 1);
    dydh = std::sqrt(v_J.x * v_J.x + v_J.y * v_J.y) /
           std::sqrt(v_B.x * v_B.x + v_B.y * v_B.y);
  } else {
    // default to returning field line length
    dydh = 1.0;
  }
}
template <typename Metric>
field_integrator<Metric>::field_integrator()
    : m_grid(), m_E(m_grid), m_B(m_grid), m_J(m_grid) {}

template <typename Metric>
void
field_integrator<Metric>::load_file(const silo_file& file) {
  if (!file.is_open()) {
    std::cerr << "File not open yet!" << std::endl;
  } else {
    m_grid = Grid(file.grid_conf());
    // TODO: how to pass in metric info
    m_grid.setup_metric(Metric{}, m_grid);

    m_E = cu_vector_field<float>(m_grid);
    m_E.set_field_type(FieldType::E);
    m_B = cu_vector_field<float>(m_grid);
    m_B.set_field_type(FieldType::B);
    m_J = cu_vector_field<float>(m_grid);
    m_J.set_field_type(FieldType::E);

    if (file.find_var("E1"))
      m_E.data(0).copy_from(file.get_multi_var("E1"));
    if (file.find_var("E2"))
      m_E.data(1).copy_from(file.get_multi_var("E2"));
    if (file.find_var("E3"))
      m_E.data(2).copy_from(file.get_multi_var("E3"));
    if (file.find_var("B1"))
      m_B.data(0).copy_from(file.get_multi_var("B1"));
    if (file.find_var("B2"))
      m_B.data(1).copy_from(file.get_multi_var("B2"));
    if (file.find_var("B3"))
      m_B.data(2).copy_from(file.get_multi_var("B3"));
    if (file.find_var("J1"))
      m_J.data(0).copy_from(file.get_multi_var("J1"));
    if (file.find_var("J2"))
      m_J.data(1).copy_from(file.get_multi_var("J2"));
    if (file.find_var("J3"))
      m_J.data(2).copy_from(file.get_multi_var("J3"));
  }
}

template <typename Metric>
template <typename Float_t>
void
field_integrator<Metric>::integrate_across_flux(
    const std::string& quantity, int num_samples,
    std::vector<Float_t>& values) {
  RungeKutta<double, double> rk(false, 0.1, true);
  field_func<Metric> f(m_E, m_B, m_J, Metric{}, quantity);
  values.resize(num_samples);

  for (int i = 0; i < num_samples; i++) {
    double theta = (i + 0.5) * CONST_PI * 0.5 / num_samples;
    Vec3<double> start(
        m_grid.mesh().lower[0] + 5 * m_grid.mesh().delta[0], theta,
        0.0);
    // start_points[i] = start;
    rk.integrate_along_field_simple(f, m_grid.mesh(), start, 0.1);
    // rk.integrate_along_field(f, m_grid.mesh(), start);
    // if (rk.output_points().back()[0] > 3.0) continue;
    // points[i] = rk.output_points();
    values[i] = rk.output_y().back();
  }
}

template <typename Metric>
template <typename Float_t>
void
field_integrator<Metric>::integrate_line(
    const std::string& quantity, const Vec3<double>& start_pos,
    double step_size, std::vector<Vec3<double>>& points,
    std::vector<Float_t>& values) {
  RungeKutta<double, double> rk(false, 0.1, true);
  field_func<Metric> f(m_E, m_B, m_J, Metric{}, quantity);

  rk.integrate_along_field_simple(f, m_grid.mesh(), start_pos,
                                  step_size);
  points = rk.output_points();
  values.resize(rk.output_y().size());
  for (size_t i = 0; i < rk.output_y().size(); i++) {
    values[i] = rk.output_y()[i];
  }
}

template <typename Metric>
double
field_integrator<Metric>::sample(const std::string& quantity,
                                 const Vec3<double>& pos) {
  return 0.0;
}

}  // namespace Aperture

#endif  // _INTEGRATOR_IMPL_H_
