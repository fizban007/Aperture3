#include "pitchptr.cuh"
#include <cstdint>

namespace Aperture {

template struct pitchptr<char>;
template struct pitchptr<float>;
template struct pitchptr<double>;
template struct pitchptr<uint32_t>;
template struct pitchptr<uint64_t>;
template struct pitchptr<int>;

}
