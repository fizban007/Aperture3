# if (CUDA_FOUND)
add_definitions(-DCUDA_ENABLED)
# include_directories(${CUDA_INCLUDE_DIRS})
# message("Cuda library found, enabling the GPU part of the code.")
# else()
  # message("Cuda library not found, skipping the GPU part of the code.")
# endif()

set(Aperture_src
  "utils/logger.cpp" "utils/timer.cpp"
  "utils/hdf_exporter.cpp"
  "utils/mpi_helper.cpp"
  "data/grid.cpp" "data/grid_log_sph.cpp"
  "data/detail/particle_data_impl.cpp"
  "commandline_args.cpp" "config_file.cpp"
  "sim_environment.cpp"
  "pic_sim.cpp"
  "data/array.cu" "data/multi_array.cu" "data/fields.cu" "data/photons.cu"
  "data/particles.cu"
  # "data/particles_1d.cu"
  "data/photons.cu"
  # "data/photons_1d.cu"
  "data/fields_utils.cu"
  "algorithms/field_solver_default.cu"
  "algorithms/field_solver_log_sph.cu"
  "algorithms/ptc_pusher_constE.cu"
  # "algorithms/ptc_pusher_beadonwire.cu"
  # "algorithms/ptc_pusher_geodesic.cpp"
  "sim_data.cu"
  "ptc_updater.cu" "ptc_updater_logsph.cu"
  "cuda/constant_mem.cu" "cuda/kernels.cu"
  "radiation/radiation_transfer.cu"
  # "algorithms/current_deposit_Esirkepov.cu"
  "algorithms/finite_diff.cu"
  "algorithms/field_solver_force_free.cu"
  "algorithms/field_solver_ffe_cyl.cu"
  )
# add_library(Aperture ${Aperture_src})
add_cuda_target(Aperture ${Aperture_src})
target_link_libraries(Aperture ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt ${CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES})

# set(AperCuda_src
#   "utils/logger.cpp" "utils/timer.cpp" "utils/hdf_exporter.cpp"
#   "utils/mpi_helper.cpp"
#   "data/grid.cpp" "data/detail/particle_data_impl.cpp"
#   "commandline_args.cpp" "config_file.cpp"
#   # "sim_environment.cpp"
#   "pic_sim.cpp"
#   "data/array.cu" "data/multi_array.cu" "data/fields.cu" "data/photons.cu"
#   "data/particles.cu" "data/particles_1d.cu"
#   "data/photons.cu" "data/photons_1d.cu"
#   "data/fields_utils.cu"
#   "algorithms/field_solver_default.cu"
#   # "algorithms/ptc_pusher_beadonwire.cu"
#   "sim_environment.cu"
#   "sim_data.cu"
#   "cuda/constant_mem.cu" "cuda/kernels.cu"
#   "radiation/radiation_transfer.cu"
#   # "algorithms/ptc_pusher_beadonwire.cu"
#   # "algorithms/current_deposit_Esirkepov.cu"
#   "algorithms/finite_diff.cu" "algorithms/field_solver_force_free.cu"
#   # "radiation/inverse_compton.cu"
#   # "algorithms/current_deposit_Esirkepov_cpu.cpp"
#   # "commandline_args.cpp" "config_file.cpp" "sim_data.cpp" "sim_environment.cpp" "pic_sim.cpp" "domain_communicator.cpp"
#   # "pic_sim.cpp" "boundary_conditions.cpp"
#   # "data/multi_array.cpp" "data/grid.cpp" "data/fields.cpp" "data/particles.cpp" "data/photons.cpp"
# #   # "algorithms/finite_diff.cpp" "algorithms/field_solver_finite_diff.cpp" "algorithms/field_solver_integral.cpp"
# #   "algorithms/field_solver_integral.cpp" "algorithms/current_deposit_Esirkepov.cpp" "algorithms/ptc_pusher_geodesic.cpp" "algorithms/solve.cpp"
# #   # "algorithms/ptc_pusher_mapping.cpp"
# # "algorithms/field_solver_integral.cpp" "algorithms/current_deposit_Esirkepov.cpp" "algorithms/ptc_pusher_geodesic.cpp" "algorithms/functions.cpp"
# # "utils/hdf_exporter.cpp" "utils/mpi_comm.cpp" "utils/mpi_helper.cpp"
# # "utils/data_exporter.cpp" "utils/silo_file.cpp" "utils/mpi_helper.cpp" "utils/mpi_comm.cpp" "utils/memory.cpp"
# #   # "initial_conditions/initial_condition_dipole.cpp" "initial_conditions/initial_condition_reload.cpp" "initial_conditions/initial_condition_empty.cpp"
# #   "initial_conditions/initial_condition_wald.cpp" "initial_conditions/initial_condition_split_monopole.cpp"
# #   "boundary_conditions/fieldBC_damping.cpp" "boundary_conditions/fieldBC_coordinate.cpp" "boundary_conditions/fieldBC_conductor.cpp"
# #   "tri_mesh/mesh.cpp" "tri_mesh/triangulation.cpp" "tri_mesh/cell.cpp" "tri_mesh/fetd.cpp")
# )
# message(${HDF5_LIBRARIES})
# add_library(Aperture STATIC ${Aperture_src})
# add_cuda_target(AperCuda)
# target_link_libraries(AperCuda Aperture)

# if (CUDA_FOUND)
#   target_link_libraries(Pulsar ${CUDA_LIBRARIES})
# endif()

# if(MPI_COMPILE_FLAGS)
#   set_target_properties(Pulsar PROPERTIES
#     COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
# endif()

# if(MPI_LINK_FLAGS)
#   set_target_properties(Pulsar PROPERTIES
#     LINK_FLAGS "${MPI_LINK_FLAGS}")
# endif()

# add_subdirectory(cuda)

# add_executable(aperture "main.cpp")
# target_link_libraries(aperture Aperture AperCuda)

add_executable(exponent "exponent.cpp")
target_link_libraries(exponent Aperture)

add_executable(aper3d "main3d.cpp")
target_link_libraries(aper3d Aperture)

add_executable(aperffe "main_2dffe.cpp")
target_link_libraries(aperffe Aperture)

add_executable(aper2d "main_2dsph.cpp")
target_link_libraries(aper2d Aperture)
# add_executable(aperture3D "main3D.cpp")
# target_link_libraries(aperture3D Pulsar)

# if (CUDA_FOUND)
#   target_link_libraries(aperture Pulsar_dev)
# endif()
