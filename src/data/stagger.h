#ifndef _STAGGER_H_
#define _STAGGER_H_

#include "cuda/cuda_control.h"

namespace Aperture {

class Stagger {
 private:
  unsigned char stagger;

 public:
  HOST_DEVICE Stagger() : stagger(0) {}
  HOST_DEVICE Stagger(unsigned char s) : stagger(s) {};
  HOST_DEVICE Stagger(const Stagger& s) :
      stagger(s.stagger) {}

  HD_INLINE Stagger& operator=(const Stagger& s) {
    stagger = s.stagger;
    return *this;
  }

  HD_INLINE Stagger& operator=(const unsigned char s) {
    stagger = s;
    return *this;
  }

  HD_INLINE int operator[](int i) const {
    return (stagger >> i) & 1UL;
  }

  HD_INLINE void set_bit(int bit, bool i) {
    unsigned long x = !!i;
    stagger ^= (-x ^ stagger) & (1UL << bit);
  }

  HD_INLINE unsigned char& data() {
    return stagger;
  }

  HD_INLINE unsigned char data() const {
    return stagger;
  }

  HD_INLINE void flip(int n) {
    stagger ^= (1UL << n);
  }

};

}

#endif  // _STAGGER_H_
