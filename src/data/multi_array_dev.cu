#include "data/detail/multi_array_dev_impl.hpp"
// #include "data/detail/multi_array_iter_impl.hpp"

namespace Aperture {

template class multi_array_dev<long long>;
template class multi_array_dev<long>;
template class multi_array_dev<int>;
template class multi_array_dev<short>;
template class multi_array_dev<char>;
template class multi_array_dev<unsigned int>;
template class multi_array_dev<unsigned long>;
template class multi_array_dev<unsigned long long>;
template class multi_array_dev<float>;
template class multi_array_dev<double>;
template class multi_array_dev<long double>;

}  // namespace Aperture
