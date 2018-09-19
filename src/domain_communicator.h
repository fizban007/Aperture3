#ifndef _DOMAIN_COMMUNICATOR_H_
#define _DOMAIN_COMMUNICATOR_H_

#include "data/fields.h"
#include "data/multi_array.h"
#include "data/particle_base.h"
#include "sim_environment.h"

namespace Aperture {

class DomainCommunicator {
 public:
  typedef MultiArray<double> array_t;
  typedef VectorField<Scalar> vec_field_t;
  typedef ScalarField<Scalar> sca_field_t;

  DomainCommunicator(Environment& env);
  ~DomainCommunicator();

  template <typename ParticleClass>
  void send_recv_particles(ParticleBase<ParticleClass>& particles,
                           const Grid& grid);

  void get_guard_cells(vec_field_t& field);
  void get_guard_cells(sca_field_t& field);
  template <typename T>
  void get_guard_cells(MultiArray<T>& array, const Grid& grid);

  void put_guard_cells(vec_field_t& field);
  void put_guard_cells(sca_field_t& field);
  template <typename T>
  void put_guard_cells(MultiArray<T>& array, const Grid& grid,
                       int stagger = 0);

 private:
  template <typename T>
  void get_guard_cells_leftright(int dir, MultiArray<T>& array,
                                 CommTags leftright, const Grid& grid);

  template <typename T>
  void put_guard_cells_leftright(int dir, MultiArray<T>& array,
                                 CommTags leftright, const Grid& grid,
                                 int stagger = 0);

  template <typename ParticleClass>
  void send_particles_directional(
      ParticleBase<ParticleClass>& particles, const Grid& grid,
      int direction);

  // These are helper functions that discriminate which communication
  // buffer to get according to data array type
  std::array<std::vector<single_particle_t>, NUM_PTC_BUFFERS>&
  get_buffer(const ParticleBase<single_particle_t>& particles) {
    return m_ptc_buffers;
  };
  std::array<std::vector<single_photon_t>, NUM_PTC_BUFFERS>& get_buffer(
      const ParticleBase<single_photon_t>& particles) {
    return m_photon_buffers;
  };
  std::array<int, NUM_PTC_BUFFERS>& get_buffer_num(
      const ParticleBase<single_particle_t>& particles) {
    return m_ptc_buf_num;
  }
  std::array<int, NUM_PTC_BUFFERS>& get_buffer_num(
      const ParticleBase<single_photon_t>& particles) {
    return m_photon_buf_num;
  }

  Environment& m_env;

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
