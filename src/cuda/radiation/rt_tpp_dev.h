#ifndef _RT_TPP_DEV_H_
#define _RT_TPP_DEV_H_

namespace Aperture {

namespace Kernels {

__device__ Scalar find_tpp_rate(Scalar gamma);

__device__ Scalar find_tpp_Em(Scalar gamma);

__device__ Scalar gen_tpp_Ep(Scalar gamma, curandState* state);

}  // namespace Kernels

}  // namespace Aperture

#endif  // _RT_TPP_DEV_H_
