#ifndef _AVX_UTILS_H_
#define _AVX_UTILS_H_

#include "utils/simd.h"

namespace Aperture {

// Get an integer representing particle type from a given flag
template <typename VUI>
inline typename simd::to_signed<VUI>::type
get_ptc_type(const VUI& flag) {
  return (typename simd::to_signed<VUI>::type)(flag >> (uint32_t)29);
}


} // namespace Aperture

#endif  // _AVX_UTILS_H_
