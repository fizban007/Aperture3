#include "data/stagger.h"
#include "catch.hpp"

using namespace Aperture;

TEST_CASE("Bitwise operations on stagger", "[Stagger]") {
  Stagger s;

  s.set_bit(2, true);
  CHECK(s.data() == (unsigned char)4);
  s.set_bit(1, true);
  CHECK(s.data() == (unsigned char)6);
  s.set_bit(2, false);

  CHECK(s.data() == (unsigned char)2);
  CHECK(s[1] == 1);
  CHECK(s[0] == 0);
  CHECK(s[2] == 0);

  s = (unsigned char)0b000;
  s.flip(0); s.flip(1); s.flip(2);
  CHECK(s.data() == (unsigned char)7);
}
