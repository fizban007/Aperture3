#include "typed_pitchedptr.cuh"
#include <cstdint>

namespace Aperture {

template struct typed_pitchedptr<char>;
template struct typed_pitchedptr<float>;
template struct typed_pitchedptr<double>;
template struct typed_pitchedptr<uint32_t>;
template struct typed_pitchedptr<uint64_t>;

}
