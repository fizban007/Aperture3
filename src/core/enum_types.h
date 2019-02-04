#ifndef _ENUM_TYPES_H_
#define _ENUM_TYPES_H_

#include <cstdint>
// #include <bitset>

// #define PARTICLE_TYPE_NUM 3

namespace Aperture {

/// Field staggering type
enum class FieldType : char { E, B };

/// Field normalization, not used
enum class FieldNormalization : unsigned char {
  coord,
  physical,
  volume
};

/// Particle types 
enum class ParticleType : unsigned char { electron = 0, positron, ion };

enum class CommTags : char { left = 0, right };

enum class Zone : char { center = 13 };

enum class BoundaryPos : char {
  lower0,
  upper0,
  lower1,
  upper1,
  lower2,
  upper2
};

enum class ParticleBCType {
  // inject,    ///< Inject particles at the boundary
  outflow,  ///< Allow particles to outflow and decouple from the fields
  reflect,  ///< Particles reflect from the boundary
  periodic,  ///< Set to this if periodic boundary condition is true
  nothing    ///< Do nothing special to the particles at the boundary
};           // ----- end of enum ParticleBCType -----

enum class FieldBCType {
  conductor,
  rotating_conductor,
  velocity_field,
  damping,
  coordinate,
  external,
  function,
  function_split,
  periodic,
  nothing
};  // ----- end of enum FieldBCType -----

// Use util functions check_bit, set_bit, bit_or, clear_bit, and
// toggle_bit to interact with particle and photon flags. These are
// defined from lower bits.
enum class ParticleFlag : uint32_t {
  tracked = 1,
  ignore_force,
  ignore_current,
  ignore_EM,
  ignore_radiation,
  primary,
  secondary,
  existed,
  annihilate
};

enum class PhotonFlag : uint32_t { tracked = 1, ignore_pair_create };

///  Composite structure that contains the information for both
///  particle boundary condition and field boundary condition
struct BCType {
  ParticleBCType ptcBC;
  FieldBCType fieldBC;
};

enum class FieldInterpolationFlag { contravariant, covariant };

/// There are three processor states. Primary means the process
/// participates in the Cartesian domain decomposition; Replica means
/// the process is a slave to a primary process and only provides
/// computing power to process the particles; Idle means the process
/// does not participate in the computation at all.
enum class ProcessState : int { primary = 0, replica, idle };

// ensemble adjustment code that indicates how a process behaves during
// the dynamic adjustment during one of the shrinking and expanding
// phases
enum class EnsAdjustCode : int {
  stay = 0,  // stay in the ensemble, whether shrinking or expanding
  move,      // leave a shrinking ensemble or join an expanding one
  idle,      // does not participate in the current phase
};

////// Magic numbers start here
enum : unsigned char { CENTER_ZONE = 13 };

enum class ForceAlgorithm : char { Boris, Vay };

enum class LogLevel : char { info, detail, debug };

}  // namespace Aperture

#endif  // _ENUM_TYPES_H_
