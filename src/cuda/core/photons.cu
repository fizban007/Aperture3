#include "core/photons.h"
#include "particle_base_impl.cu"

namespace Aperture {

template class particle_base<single_photon_t>;

namespace Kernels {

// template <typename PtcData>
__global__ void
append_ph(photon_data data, size_t num, Vec3<Pos_t> x, Vec3<Scalar> p,
          Scalar path_left, int cell, Scalar w, uint32_t flag) {
  printf("%f, %f, %f\n", x[0], x[1], x[2]);
  data.x1[num] = x[0];
  data.x2[num] = x[1];
  data.x3[num] = x[2];
  data.p1[num] = p[0];
  data.p2[num] = p[1];
  data.p3[num] = p[2];
  data.path_left[num] = path_left;
  data.weight[num] = w;
  data.cell[num] = cell;
  data.flag[num] = flag;
}

}  // namespace Kernels

photons_t::photons_t() {}

photons_t::photons_t(std::size_t max_num, bool managed)
    : particle_base<single_photon_t>(max_num, managed) {}

// photons_t::photons_t(const photons_t& other)
//     : particle_base<single_photon_t>(other) {}

photons_t::photons_t(photons_t&& other)
    : particle_base<single_photon_t>(std::move(other)) {}

photons_t::~photons_t() {}

void
photons_t::append(const Vec3<Pos_t>& x, const Vec3<Scalar>& p,
                  int cell, Scalar path_left, Scalar weight,
                  uint32_t flag) {
  Kernels::append_ph<<<1, 1>>>(m_data, m_number, x, p, path_left, cell,
                               weight, flag);
  CudaCheckError();
  m_number += 1;
  cudaDeviceSynchronize();
}

}  // namespace Aperture
