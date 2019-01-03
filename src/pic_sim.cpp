#include "pic_sim.h"
#include "algorithms/current_deposit_Esirkepov.h"
#include "field_solvers.h"
#include "ptc_pushers.h"
// #include "domain_communicator.h"
#include <functional>
#include <memory>

namespace Aperture {

// PICSim::PICSim() :
//     PICSim(Environment::get_instance()) {}

PICSim::PICSim(Environment& env) : m_env(env) {
  // Logger::print_info("Periodic is {}",
  // env.conf().boundary_periodic[0]); Initialize modules m_comm =
  // std::make_unique<DomainCommunicator>(env);

  // Select current deposition method according to config
  // m_depositer = std::unique_ptr<CurrentDepositer_Esirkepov>(
  //     new CurrentDepositer_Esirkepov(m_env));
  // m_depositer->set_periodic(env.conf().boundary_periodic[0]);
  // m_depositer->set_interp_order(env.conf().interpolation_order);

  // Select field solver according to config
  if (m_env.params().algorithm_field_update == "ffe") {
    m_field_solver = std::unique_ptr<FieldSolver_FFE>(
        new FieldSolver_FFE(m_env.local_grid()));
  } else {
    // Fall back to default field solver
    m_field_solver = std::unique_ptr<FieldSolver_FFE>(
        new FieldSolver_FFE(m_env.local_grid()));
  }
  // TODO: select particle mover type according to config
  // int interp_order = m_env.conf().interpolation_order;

  // Select particle pusher according to config
  // if (m_env.params().algorithm_ptc_move == "beadonwire") {
  //   m_pusher = std::unique_ptr<ParticlePusher_BeadOnWire>(
  //       new ParticlePusher_BeadOnWire(m_env));
  // } else
  // if (m_env.params().algorithm_ptc_move == "constE") {
    // m_pusher = std::unique_ptr<ParticlePusher_ConstE>(
        // new ParticlePusher_ConstE(m_env));
  // }
  // } else {
  //   m_pusher = std::unique_ptr<ParticlePusher_Geodesic>(
  //       new ParticlePusher_Geodesic(m_env));
  // }
  // m_pusher->set_periodic(env.conf().boundary_periodic[0]);
  // m_pusher->set_interp_order(env.conf().interpolation_order);

  // Inverse Compton module
  // m_inverse_compton = std::unique_ptr<InverseCompton>(new
  // InverseCompton(m_env));

  // TODO: figure out a way to set algorithm
  // if (m_env.conf().algorithm_ptc_push == "Vay")
  //   m_pusher -> set_algorithm(ForceAlgorithm::Vay);
  // else if (m_env.conf().algorithm_ptc_push == "Boris")
  //   m_pusher -> set_algorithm(ForceAlgorithm::Boris);
  // m_pusher -> set_gravity(m_env.conf().gravity);
  // m_pusher -> set_radiation(bool radiation)

  // Register communication callbacks
  // m_depositer->register_current_callback(
  //     [this](cu_vector_field<Scalar>& j) { m_comm->put_guard_cells(j);
  //     });

  // m_depositer->register_rho_callback(
  //     [this](cu_scalar_field<Scalar>& rho) {
  //     m_comm->put_guard_cells(rho); });

  // m_field_solver->register_comm_callback(
  //     [this](cu_vector_field<Scalar>& f) -> void {
  //     m_comm->get_guard_cells(f); });

  // m_field_solver->register_comm_callback(
  //     [this](cu_scalar_field<Scalar>& f) -> void {
  //     m_comm->get_guard_cells(f); });

  // auto &comm = *m_comm;
  // std::function<void(cu_vector_field<Scalar>&)> vcall =
  // [&comm](cu_vector_field<Scalar>& f) -> void { comm.get_guard_cells(f);
  // }; m_field_solver->register_comm_callback(std::bind(
  //     static_cast<void(DomainCommunicator::*)(cu_vector_field<Scalar>&)>(&DomainCommunicator::get_guard_cells),
  //     &comm));
  // m_field_solver->register_communicator(m_comm.get());

  // Register boundary condition
  // m_field_solver->set_boundary_condition(env.boundary_conditions());
}

PICSim::~PICSim() {}

void
PICSim::loop(Aperture::cu_sim_data& data, uint32_t steps,
             uint32_t data_freq) {
  for (uint32_t n = 0; n < steps; n++) {
    // Do stuff
    step(data, n);
  }
}

void
PICSim::step(Aperture::cu_sim_data& data, uint32_t step) {
  double dt = m_env.params().delta_t;
  // Particle push, move, and photon move are all handled here
  // m_pusher->push(data, dt);
  // m_inverse_compton->convert_pairs(data.particles, data.photons);
  // m_depositer->deposit(data, dt);
  m_ptc_updater->update_particles(data, dt);
  m_ptc_updater->handle_boundary(data);
  m_field_solver->update_fields(data, dt);
  // m_inverse_compton->emit_photons(data.photons, data.particles);
  // data.photons.emit_photons(data.particles[0], data.particles[1],
  // data.E.grid().mesh()); data.photons.move(data.E.grid(), dt);
  // data.photons.convert_pairs(data.particles[0], data.particles[1]);

  // auto& mesh = data.E.grid().mesh();
  // Logger::print_info("J at boundary 1: {} | {} | {} | {}", data.J(0,
  // 1),
  //                    data.J(0, 2), data.J(0, 3), data.J(0, 4));
  // Logger::print_info("J at boundary 2: {} | {} | {} | {}", data.J(0,
  // 0),
  //                    data.J(0, 1), data.J(0, 2), data.J(0, 3));

  // m_pusher->handle_boundary(data);
  // Sort the particles every 20 timesteps to move empty slots to the
  // back
  if ((step % 20) == 0) {
    data.particles.sort_by_cell();
  }
  // if ((step % 200) == 0) {
  //   data.photons.sort(data.E.grid());
  // }
  Logger::print_info("There are {} particles in the pool",
                     data.particles.number());
  // Logger::print_info("There are {} positrons in the pool",
  // data.particles[1].number());

  // uint32_t total_tracked_e = 0;
  // uint32_t total_tracked_ph = 0;
  // for (Index_t idx = 0; idx < data.particles[0].number(); idx++) {
  //   if (!data.particles[0].is_empty(idx) &&
  //   data.particles[0].check_flag(idx, ParticleFlag::tracked))
  //     total_tracked_e += 1;
  // }
  // for (Index_t idx = 0; idx < data.photons.number(); idx++) {
  //   if (!data.photons.is_empty(idx) && data.photons.check_flag(idx,
  //   PhotonFlag::tracked))
  //     total_tracked_ph += 1;
  // }
  // Logger::print_info("{} electrons are tracked", total_tracked_e);
  // Logger::print_info("{} photons are tracked", total_tracked_ph);
}

}  // namespace Aperture
