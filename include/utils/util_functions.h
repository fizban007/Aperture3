#ifndef _UTIL_FUNCTIONS_H_
#define _UTIL_FUNCTIONS_H_

#include <string>
#include "data/typedefs.h"
#include "data/enum_types.h"

namespace Aperture {

template <typename T>
int
sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

template < typename PtcFlag >
inline bool check_bit( uint32_t flag, PtcFlag bit ) {
  return ( flag & (1 << static_cast<int>(bit)) ) != 0;
}

template < typename PtcFlag >
inline uint32_t bit_or( PtcFlag bit ) {
  return (1 << static_cast<int>(bit));
}

template < typename PtcFlag, typename ...P  >
inline uint32_t bit_or( PtcFlag bit, P ...bits ) {
  return ( (1 << static_cast<int>(bit)) | bit_or( bits... )  );
}

template < typename ...PtcFlag  >
inline void set_bit( uint32_t& flag, PtcFlag ...bits ) {
  flag |= bit_or( bits... );
}

template < typename ...PtcFlag  >
inline void clear_bit( uint32_t& flag, PtcFlag ...bits ) {
  flag &= ~static_cast<int>( bit_or(bits...) );
}

template < typename ...PtcFlag  >
inline void toggle_bit( uint32_t& flag, PtcFlag ...bits ) {
  flag ^= static_cast<int>( bit_or(bits...) );
}


// turn particle type into string
inline std::string NameStr( ParticleType type ) noexcept {
  if ( ParticleType::electron == type ) return "Electron";
  else if ( ParticleType::positron == type ) return "Positron";
  else if ( ParticleType::ion == type ) return "Ion";
  else return "";
}

// for uniform interface, this overload is for photons
inline std::string NameStr( const std::string& photon_type ) noexcept {
  return photon_type;
}

// flip between 0 and 1
inline int flip (int n) {
  return (n == 1 ? 0 : 1);
}

// flip between true and false
inline bool flip (bool n) {
  return !n;
}


}

#endif  // _UTIL_FUNCTIONS_H_
