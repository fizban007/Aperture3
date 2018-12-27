#ifndef _PARTICLE_INTERFACE_H_
#define _PARTICLE_INTERFACE_H_

namespace Aperture {

////////////////////////////////////////////////////////////////////////////////
///  Interface class for particle storage classes. Both particle and
///  photon classes derive from this one, regardless of whether storage
///  is on main RAM or on GPU.
////////////////////////////////////////////////////////////////////////////////
class particle_interface {
 protected:
  /// Maximum number of particles in the array, fixed after
  /// initialization
  std::size_t m_size = 0;
  /// The current number of particles in the array.
  /// @accessors #number(), #setNum()
  std::size_t m_number = 0;

 public:
  particle_interface() {}
  particle_interface(std::size_t max_num) : m_size(max_num) {}
  virtual ~particle_interface() {}

  /// @return Returns the value of the current number of particles
  std::size_t number() const { return m_number; }

  /// @return Returns the maximum number of particles
  std::size_t size() const { return m_size; }

  /// Set the current number of particles in the array to a given value
  /// @param num New number of particles in the array
  void set_num(size_t num) { m_number = num; }
};  // ----- end of class particle_interface -----

}  // namespace Aperture

#endif  // _PARTICLE_INTERFACE_H_
