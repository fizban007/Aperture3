#include "pitchptr.cuh"
#include <cstdint>

namespace Aperture {

template <typename T>
pitchptr<T>
get_pitchptr(multi_array<T>& array) {
  return pitchptr<T>(get_cudaPitchedPtr(array));
}

template <typename T>
cudaPitchedPtr
get_cudaPitchedPtr(multi_array<T>& array) {
  return make_cudaPitchedPtr(array.dev_ptr(), array.pitch(),
                             array.extent().width(),
                             array.extent().height());
}

template struct pitchptr<char>;
template struct pitchptr<float>;
template struct pitchptr<double>;
template struct pitchptr<uint32_t>;
template struct pitchptr<uint64_t>;
template struct pitchptr<int>;

}  // namespace Aperture
