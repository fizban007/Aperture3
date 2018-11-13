#ifndef _UTIL_FUNCTIONS_H_
#define _UTIL_FUNCTIONS_H_

#include "cuda/cuda_control.h"
#include "data/enum_types.h"
#include "data/typedefs.h"
#include <string>

namespace Aperture {

HD_INLINE Scalar*
ptrAddr(cudaPitchedPtr p, size_t offset) {
  return (Scalar*)((char*)p.ptr + offset);
}

HD_INLINE double*
ptrAddr_d(cudaPitchedPtr p, size_t offset) {
  return (double*)((char*)p.ptr + offset);
}

template <typename T>
HD_INLINE T
square(const T &val) {
  return val * val;
}

template <typename T>
HD_INLINE int
sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename PtcFlag>
HD_INLINE bool
check_bit(uint32_t flag, PtcFlag bit) {
  return (flag & (1 << static_cast<int>(bit))) != 0;
}

template <typename PtcFlag>
HD_INLINE uint32_t
bit_or(PtcFlag bit) {
  return (1 << static_cast<int>(bit));
}

template <typename PtcFlag, typename... P>
HD_INLINE uint32_t
bit_or(PtcFlag bit, P... bits) {
  return ((1 << static_cast<int>(bit)) | bit_or(bits...));
}

template <typename... PtcFlag>
HD_INLINE void
set_bit(uint32_t& flag, PtcFlag... bits) {
  flag |= bit_or(bits...);
}

template <typename... PtcFlag>
HD_INLINE void
clear_bit(uint32_t& flag, PtcFlag... bits) {
  flag &= ~static_cast<int>(bit_or(bits...));
}

template <typename... PtcFlag>
HD_INLINE void
toggle_bit(uint32_t& flag, PtcFlag... bits) {
  flag ^= static_cast<int>(bit_or(bits...));
}

// Get an integer representing particle type from a given flag
HD_INLINE int
get_ptc_type(uint32_t flag) {
  return (int)(flag >> 29);
}

// Generate a particle flag from a give particle type
HD_INLINE uint32_t
gen_ptc_type_flag(ParticleType type) {
  return ((uint32_t)type << 29);
}

// Set a given flag such that it now represents given particle type
HD_INLINE uint32_t
set_ptc_type_flag(uint32_t flag, ParticleType type) {
  return (flag & ((uint32_t)-1 >> 3)) | gen_ptc_type_flag(type);
}

// turn particle type into string
inline std::string
NameStr(ParticleType type) noexcept {
  if (ParticleType::electron == type)
    return "Electron";
  else if (ParticleType::positron == type)
    return "Positron";
  else if (ParticleType::ion == type)
    return "Ion";
  else
    return "";
}

// for uniform interface, this overload is for photons
inline std::string
NameStr(const std::string& photon_type) noexcept {
  return photon_type;
}

// flip between 0 and 1
HD_INLINE int
flip(int n) {
  return (n == 1 ? 0 : 1);
}

// flip between true and false
HD_INLINE bool
flip(bool n) {
  return !n;
}

}  // namespace Aperture

#endif  // _UTIL_FUNCTIONS_H_
