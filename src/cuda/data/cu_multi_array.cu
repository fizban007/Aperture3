#include "cuda/data/detail/cu_multi_array_impl.hpp"
// #include "core/detail/multi_array_iter_impl.hpp"

namespace Aperture {

template class cu_multi_array<long long>;
template class cu_multi_array<long>;
template class cu_multi_array<int>;
template class cu_multi_array<short>;
template class cu_multi_array<char>;
template class cu_multi_array<unsigned int>;
template class cu_multi_array<unsigned long>;
template class cu_multi_array<unsigned long long>;
template class cu_multi_array<float>;
template class cu_multi_array<double>;
template class cu_multi_array<long double>;

}  // namespace Aperture
