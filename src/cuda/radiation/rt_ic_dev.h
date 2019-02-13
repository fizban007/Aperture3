#include "core/typedefs.h"

namespace Aperture {

__device__ int find_n_gamma(Scalar gamma);

__device__ int find_n_e1(Scalar e1);

__device__ int binary_search(float u, Scalar* array, int size,
                             Scalar& l, Scalar& h);

__device__ int binary_search(float u, int n, cudaPitchedPtr array,
                             Scalar& l, Scalar& h);

__device__ Scalar gen_photon_e(Scalar gamma, curandState* state);

__device__ Scalar find_ic_rate(Scalar gamma);

__device__ Scalar find_gg_rate(Scalar eph);

}  // namespace Aperture
