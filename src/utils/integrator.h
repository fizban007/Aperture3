#ifndef _INTEGRATOR_H_
#define _INTEGRATOR_H_

#include <string>
#include "data/grid.h"
#include "data/vec3.h"
#include "utils/silo_file.h"
#include "algorithms/runge_kutta.h"

namespace Aperture {

template <typename Metric>
struct field_func {
  const VectorField<float>& E;
  const VectorField<float>& B;
  const VectorField<float>& J;
  Metric g;
  std::string component;

  field_func(const VectorField<float>& Efield,
             const VectorField<float>& Bfield,
             const VectorField<float>& Jfield,
             const Metric& metric, const std::string& comp)
      : E(Efield), B(Bfield), J(Jfield), g(metric), component(comp) {}

  void operator() (const Vec3<double>& pos, Vec3<double>& v,
                   double& dydh) const;
};

template <typename Metric>
class field_integrator
{
 public:
  field_integrator();
  ~field_integrator() {}

  // void load_file(const std::string& filename);
  void load_file(const silo_file& file);

  template <typename Float_t>
  void integrate_line(const std::string& quantity, const Vec3<double>& start_pos, double step_size,
                      std::vector<Vec3<double> >& points, std::vector<Float_t>& values);

  template <typename Float_t>
  void integrate_across_flux(const std::string& quantity, int num_samples, std::vector<Float_t>& values);

  double sample(const std::string& quantity, const Vec3<double>& pos);

 private:
  Grid m_grid;
  VectorField<float> m_E, m_B, m_J;
  // RungeKutta<double, double> m_rk;
}; // ----- end of class field_integrator -----

}

#include "utils/integrator_impl.hpp"

#endif  // _INTEGRATOR_H_
