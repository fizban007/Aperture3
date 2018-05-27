#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include "fadiff.h"
#include "badiff.h"
// #include "algorithms/interp_template.h"
#include "catch.hpp"
// #include "grid.h"
#include <chrono>
#include <unordered_map>

// using namespace Aperture;
using namespace fadbad;
using namespace std::chrono;

std::unordered_map<std::string, high_resolution_clock::time_point> t_stamps;
high_resolution_clock::time_point t_now = high_resolution_clock::now();

void
stamp(const std::string& name = "") {
  t_stamps[name] = high_resolution_clock::now();
}

void
show_duration_since_stamp(const std::string& routine_name, const std::string& unit, const std::string& stamp_name = "") {
  t_now = high_resolution_clock::now();
  if (routine_name == "" && stamp_name == "") {
    std::cout << "--- Time for default clock is ";
  } else if (routine_name == ""){
    std::cout << "--- Time for " << stamp_name << " is ";
  } else {
    std::cout << "--- Time for " << routine_name << " is ";
  }
  if (unit == "second" || unit == "s") {
    auto dur = duration_cast<duration<float, std::ratio<1, 1> > >(t_now - t_stamps[stamp_name]);
    std::cout << dur.count() << "s" << std::endl;
  } else if (unit == "millisecond" || unit == "ms") {
    auto dur = duration_cast<milliseconds>(t_now - t_stamps[stamp_name]);
    std::cout << dur.count() << "ms" << std::endl;
  } else if (unit == "microsecond" || unit == "us") {
    auto dur = duration_cast<microseconds>(t_now - t_stamps[stamp_name]);
    std::cout << dur.count() << "µs" << std::endl;
  } else if (unit == "nanosecond" || unit == "ns") {
    auto dur = duration_cast<nanoseconds>(t_now - t_stamps[stamp_name]);
    std::cout << dur.count() << "ns" << std::endl;
  }
}

template <typename Double>
Double func2(Double x, Double y) {
  Double result = x + y;
  return result;
}

class Func
{
 public:
  Func(double rg, double omega, double theta) :
      rg_(rg), omega_(omega), theta_(theta) {
    s_ = std::sin(theta_);
  }

  ~Func() {}

  template <typename Double>
  Double operator()(Double r, double vr) {
    Double f = (1.0 - rg_ / r);
    return sqrt(f - (vr*vr/f + r*r*s_*s_*omega_*omega_*(1.0 - r*s_*vr)*(1.0 - r*s_*vr)));
  }

  template <typename Double>
  Double dLdr(Double r, Double vr) {
    Double result = rg_*(1.0 / (r*r) + vr*vr/(r - rg_)/(r - rg_));
    result -= 2.0 * r * omega_ * omega_ * s_ * s_;
    result += 6.0 * r * r * vr * omega_ * omega_ * s_ * s_ * s_;
    result -= 4.0 * r * r * r * vr * vr * omega_ * omega_ * s_ * s_ * s_ * s_;
    return result;
  }

 private:
  double rg_, omega_, theta_;
  double s_;
}; // ----- end of class Func -----


TEST_CASE("Complex compounded derivatives with intermediate values", "[derivative]") {
  F<double> x, y, f;
  x = 0.3; x.diff(0, 2);
  y = 0.4; y.diff(1, 2);
  F<double> g = x * x - sqrt(1.0 + x * x + y * y);
  f = (x * x + y * y) / g;

  double fval = f.x();
  double dfdx = f.d(0);
  double dfdy = f.d(1);

  std::cout << "f(x) = " << fval << std::endl;
  std::cout << "dfdx(x) = " << dfdx << std::endl;
  std::cout << "dfdy(x) = " << dfdy << std::endl;
}

TEST_CASE("Convert from double", "[AD]") {
  typedef F<double> var;
  var a = func2<var>(1.0, 2.0);
  std::cout << a.x() << std::endl;
}

TEST_CASE("Model Lagrangian in Schwarzschild", "[AD]") {
  Func L(2.0, 0.001, 0.4);
  typedef F<double, 2> var;
  typedef B<double> var2;
  var l = L(var(1.5), 0.99);
  std::cout << l.x() << std::endl;

  std::random_device dev;
  std::mt19937_64 eng;
  eng.seed(dev());
  std::uniform_real_distribution<float> distribution(5.0, 10.5);
  std::uniform_real_distribution<float> dist_v(0.0, 0.1);

  const int N = 5000000;
  std::vector<double> vec(N);
  std::vector<double> v_r(N);
  for (size_t i = 0; i < N; ++i) {
    float tmp = distribution(eng);
    vec[i] = tmp;
    v_r[i] = dist_v(eng);
  }

  stamp();
  for (int i = 0; i < N; i++) {
    double res = L(vec[i], v_r[i]);
    // std::cout << vec[i] << ", " << v_r[i] << ", " << res << std::endl;
  }
  show_duration_since_stamp("For double", "ms");
  F<double, 1> x, y, res;
  stamp();
  for (int i = 0; i < N; i++) {
    x = vec[i]; x.diff(0);
    // y = v_r[i]; y.diff(1);
    res = L(x, v_r[i]);
    double dLdr = res.d(0);
    // double l = res.x();
    // if (l != l)
    //   std::cout << "NaN at " << vec[i] << std::endl;
    // else
    //   CHECK(L.dLdr(vec[i], v_r[i])*0.5/res.x() == Approx(dLdr));
  }
  show_duration_since_stamp("For AD", "ms");

  stamp();
  for (int i = 0; i < N; i++) {
    double dLdr = L.dLdr(vec[i], v_r[i]) * 0.5 / L(vec[i], v_r[i]);
  }
  show_duration_since_stamp("For analytic", "ms");
}
