#ifndef _INTEGRATOR_H_
#define _INTEGRATOR_H_

#include "core/runge_kutta.h"
#include "core/grid.h"
#include "core/vec3.h"
#include "utils/silo_file.h"
#include <string>

namespace Aperture {

template <typename Metric>
struct field_func {
  const cu_vector_field<float>& E;
  const cu_vector_field<float>& B;
  const cu_vector_field<float>& J;
  Metric g;
  std::string component;

  field_func(const cu_vector_field<float>& Efield,
             const cu_vector_field<float>& Bfield,
             const cu_vector_field<float>& Jfield, const Metric& metric,
             const std::string& comp)
      : E(Efield), B(Bfield), J(Jfield), g(metric), component(comp) {}

  void operator()(const Vec3<double>& pos, Vec3<double>& v,
                  double& dydh) const;
};

template <typename Metric>
class field_integrator {
 public:
  field_integrator();
  ~field_integrator() {}

  // void load_file(const std::string& filename);
  void load_file(const silo_file& file);

  template <typename Float_t>
  void integrate_line(const std::string& quantity,
                      const Vec3<double>& start_pos, double step_size,
                      std::vector<Vec3<double>>& points,
                      std::vector<Float_t>& values);

  template <typename Float_t>
  void integrate_across_flux(const std::string& quantity,
                             int num_samples,
                             std::vector<Float_t>& values);

  double sample(const std::string& quantity, const Vec3<double>& pos);

 private:
  Grid m_grid;
  cu_vector_field<float> m_E, m_B, m_J;
  // RungeKutta<double, double> m_rk;
};  // ----- end of class field_integrator -----

}  // namespace Aperture

#include "utils/integrator_impl.hpp"

#endif  // _INTEGRATOR_H_
