#include "core/detail/multi_array_impl.hpp"
#include "utils/simd.h"

namespace Aperture {

template class multi_array<long long>;
template class multi_array<long>;
template class multi_array<int>;
template class multi_array<short>;
template class multi_array<char>;
template class multi_array<unsigned int>;
template class multi_array<unsigned long>;
template class multi_array<unsigned long long>;
template class multi_array<float>;
template class multi_array<double>;
template class multi_array<long double>;
template class multi_array<simd::simd_buffer>;

}  // namespace Aperture
