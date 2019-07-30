#include "catch.hpp"
#include "utils/util_functions.h"
#include "utils/logger.h"

using namespace Aperture;

TEST_CASE("Testing flags", "[flags]") {
  uint32_t flag = bit_or(ParticleFlag::primary, ParticleFlag::tracked);
  CHECK(!check_bit(flag, ParticleFlag::ignore_EM));

  Logger::print_info("{}", bit_or(ParticleFlag::primary));
  Logger::print_info("{}", bit_or(ParticleFlag::tracked));
  Logger::print_info("{}", bit_or(ParticleFlag::ignore_force));
  Logger::print_info("{}", bit_or(ParticleFlag::ignore_current));
  Logger::print_info("{}", bit_or(ParticleFlag::ignore_EM));
  Logger::print_info("{}", bit_or(ParticleFlag::ignore_radiation));
  Logger::print_info("{}", bit_or(ParticleFlag::secondary));
  Logger::print_info("{}", bit_or(ParticleFlag::emit_photon));
}
