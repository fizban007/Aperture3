#include "core/typedefs.h"

namespace Aperture {

namespace Kernels {

extern __constant__ Scalar dev_ic_dep;
extern __constant__ Scalar dev_ic_dg;
extern __constant__ Scalar dev_ic_dlep;
extern __constant__ cudaPitchedPtr dev_ic_dNde;
extern __constant__ cudaPitchedPtr dev_ic_dNde_thompson;
extern __constant__ Scalar* dev_ic_rate;
extern __constant__ Scalar* dev_gg_rate;
extern __constant__ Scalar* dev_gammas;

__device__ int find_n_gamma(Scalar gamma);

__device__ int find_n_e1(Scalar e1);

__device__ int binary_search(float u, Scalar* array, int size,
                             Scalar& l, Scalar& h);

__device__ int binary_search(float u, int n, cudaPitchedPtr array,
                             Scalar& l, Scalar& h);

__device__ Scalar gen_photon_e(Scalar gamma, curandState* state);

__device__ Scalar find_ic_rate(Scalar gamma);

__device__ Scalar find_gg_rate(Scalar eph);

}

}  // namespace Aperture
