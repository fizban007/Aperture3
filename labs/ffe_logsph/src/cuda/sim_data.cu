#include "cuda/constant_mem.h"
#include "cuda/constant_mem_func.h"
#include "cuda/cudaUtility.h"
#include "cuda/data_ptrs.h"
#include "cuda/kernels.h"
#include "cuda/utils/pitchptr.h"
#include "sim_data_impl.hpp"
#include "visit_struct/visit_struct.hpp"

namespace Aperture {

namespace Kernels {

template <typename T>
__global__ void
compute_EdotB(pitchptr<T> e1, pitchptr<T> e2, pitchptr<T> e3,
              pitchptr<T> b1, pitchptr<T> b2, pitchptr<T> b3,
              pitchptr<T> b1bg, pitchptr<T> b2bg, pitchptr<T> b3bg,
              pitchptr<T> EdotB) {
  // Compute time-averaged EdotB over the output interval
  int t1 = blockIdx.x, t2 = blockIdx.y;
  int c1 = threadIdx.x, c2 = threadIdx.y;
  int n1 = dev_mesh.guard[0] + t1 * blockDim.x + c1;
  int n2 = dev_mesh.guard[1] + t2 * blockDim.y + c2;
  // size_t globalOffset = n2 * e1.pitch + n1 * sizeof(Scalar);
  size_t globalOffset = e1.compute_offset(n1, n2);

  float delta = 1.0f / dev_params.data_interval;
  Scalar E1 = 0.5f * (e1(n1, n2) + e1(n1, n2 - 1));
  Scalar E2 = 0.5f * (e2(n1, n2) + e2(n1 - 1, n2));
  Scalar E3 = 0.25f * (e3(n1, n2) + e3(n1 - 1, n2) + e3(n1, n2 - 1) +
                       e3(n1 - 1, n2 - 1));
  Scalar B1 = 0.5f * (b1(n1, n2) + b1(n1 - 1, n2)) +
              0.5f * (b1bg(n1, n2) + b1bg(n1 - 1, n2));
  Scalar B2 = 0.5f * (b2(n1, n2) + b2(n1, n2 - 1)) +
              0.5f * (b2bg(n1, n2) + b2bg(n1, n2 - 1));
  Scalar B3 = b3[globalOffset] + b3bg[globalOffset];

  // Do the actual computation here
  EdotB[globalOffset] += delta * (E1 * B1 + E2 * B2 + E3 * B3) /
                         sqrt(B1 * B1 + B2 * B2 + B3 * B3);
}

}  // namespace Kernels
void
sim_data::initialize(const sim_environment& env) {
  init_bg_fields();
}

void
sim_data::finalize() {}

void
sim_data::init_bg_fields() {}

}  // namespace Aperture
