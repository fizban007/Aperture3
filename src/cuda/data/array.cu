#include "cuda/data/detail/array_impl.hpp"

namespace Aperture {

template class cu_array<long long>;
template class cu_array<long>;
template class cu_array<int>;
template class cu_array<short>;
template class cu_array<char>;
template class cu_array<uint16_t>;
template class cu_array<uint32_t>;
template class cu_array<uint64_t>;
template class cu_array<float>;
template class cu_array<double>;
template class cu_array<long double>;

}  // namespace Aperture