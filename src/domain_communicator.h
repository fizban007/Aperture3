#ifndef _DOMAIN_COMMUNICATOR_H_
#define _DOMAIN_COMMUNICATOR_H_

#include "data/fields_dev.h"
#include "data/cu_multi_array.h"
#include "data/particle_base.h"
#include "sim_environment_dev.h"

namespace Aperture {

class DomainCommunicator {
 public:
  typedef cu_multi_array<double> array_t;
  typedef cu_vector_field<Scalar> vec_field_t;
  typedef cu_scalar_field<Scalar> sca_field_t;

  DomainCommunicator(cu_sim_environment& env);
  ~DomainCommunicator();

  template <typename ParticleClass>
  void send_recv_particles(particle_base<ParticleClass>& particles,
                           const Grid& grid);

  void get_guard_cells(vec_field_t& field);
  void get_guard_cells(sca_field_t& field);
  template <typename T>
  void get_guard_cells(cu_multi_array<T>& array, const Grid& grid);

  void put_guard_cells(vec_field_t& field);
  void put_guard_cells(sca_field_t& field);
  template <typename T>
  void put_guard_cells(cu_multi_array<T>& array, const Grid& grid,
                       int stagger = 0);

 private:
  template <typename T>
  void get_guard_cells_leftright(int dir, cu_multi_array<T>& array,
                                 CommTags leftright, const Grid& grid);

  template <typename T>
  void put_guard_cells_leftright(int dir, cu_multi_array<T>& array,
                                 CommTags leftright, const Grid& grid,
                                 int stagger = 0);

  template <typename ParticleClass>
  void send_particles_directional(
      particle_base<ParticleClass>& particles, const Grid& grid,
      int direction);

  // These are helper functions that discriminate which communication
  // buffer to get according to data array type
  std::array<std::vector<single_particle_t>, NUM_PTC_BUFFERS>&
  get_buffer(const particle_base<single_particle_t>& particles) {
    return m_ptc_buffers;
  };
  std::array<std::vector<single_photon_t>, NUM_PTC_BUFFERS>& get_buffer(
      const particle_base<single_photon_t>& particles) {
    return m_photon_buffers;
  };
  std::array<int, NUM_PTC_BUFFERS>& get_buffer_num(
      const particle_base<single_particle_t>& particles) {
    return m_ptc_buf_num;
  }
  std::array<int, NUM_PTC_BUFFERS>& get_buffer_num(
      const particle_base<single_photon_t>& particles) {
    return m_photon_buf_num;
  }

  cu_sim_environment& m_env;

  std::vector<Index_t>
      m_ptc_partition;  ///< Partition array used to mark the
                        ///  starting point of each
                        ///  communication region

  std::array<std::vector<single_particle_t>, NUM_PTC_BUFFERS>
      m_ptc_buffers;
  std::array<std::vector<single_photon_t>, NUM_PTC_BUFFERS>
      m_photon_buffers;

  std::array<int, NUM_PTC_BUFFERS>
      m_ptc_buf_num;  ///< Number of particles in each buffer
  std::array<int, NUM_PTC_BUFFERS>
      m_photon_buf_num;  ///< Number of particles in each buffer

  std::array<array_t, 3> m_field_buf_send;
  std::array<array_t, 3> m_field_buf_recv;
};  // ----- end of class domain_communicator -----
}  // namespace Aperture

#endif  // _DOMAIN_COMMUNICATOR_H_
