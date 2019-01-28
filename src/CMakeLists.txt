# if (CUDA_FOUND)
add_definitions(-DCUDA_ENABLED)
# include_directories(${CUDA_INCLUDE_DIRS})
# message("Cuda library found, enabling the GPU part of the code.")
# else()
  # message("Cuda library not found, skipping the GPU part of the code.")
# endif()

if(LAB_ROOT_DIR)
  include("${LAB_ROOT_DIR}/src/CMakeLists.txt")
  include_directories(BEFORE "${LAB_ROOT_DIR}/include")
  include_directories(BEFORE "${LAB_ROOT_DIR}/src")
  set(shadows "${LAB_SHADOWS}")
  set(cu_shadows "${LAB_CU_SHADOWS}")
  # list(FIND shadows "UserControl.cpp" Lab_User_Control_Found)
  # if(Lab_User_Control_Found LESS 0)
  #   set(Lab_User_Control_Found NO)
  # else()
  #   set(Lab_User_Control_Found YES)
  # endif()
  # list(REMOVE_ITEM shadows "UserControl.cpp")
  if (shadows)
    message("${Yellow}Invoked LAB ${LAB_NAME} from ${LAB_ROOT_DIR}, with following shadowed${ColorReset}")
  endif()
  set(Lab_Aperture_src ${shadows})
  foreach(ARG ${LAB_SHADOWS})
      message("${Yellow}    ${ARG}${ColorReset}")
  endforeach()
  if (cu_shadows)
    message("${Yellow}Invoked LAB ${LAB_NAME} from ${LAB_ROOT_DIR}, with following CUDA sources shadowed${ColorReset}")
  endif()
  set(Lab_cuAperture_src ${cu_shadows})
  foreach(ARG ${LAB_CU_SHADOWS})
      message("${Yellow}    ${ARG}${ColorReset}")
  endforeach()
else()
  message("${Yellow}No lab is invoked.${ColorReset}")
endif()

set(Aperture_src
  "utils/logger.cpp"
  "utils/timer.cpp"
  "utils/mpi_helper.cpp"
  "utils/memory.cpp"
  "utils/filesystem.cpp"
  "core/multi_array.cpp"
  "core/grid.cpp"
  "grids/grid_log_sph.cpp"
  "data/fields.cpp"
  "data/detail/particle_data_impl.cpp"
  "data/particles.cpp"
  "data/photons.cpp"
  # "utils/hdf_exporter.cpp"
  "utils/data_exporter.cpp"
  "commandline_args.cpp"
  "config_file.cpp"
  "sim_environment.cpp"
  "sim_data.cpp"
  # "pic_sim.cpp"
  "core/field_solver_default.cpp"
  "core/field_solver_1d.cpp"
  "core/ptc_updater_default.cpp"
  "core/ptc_updater_1d.cpp"
  "radiation/rt_1d.cpp"
  )

# modify Aperture_src using the files in the lab
if(LAB_ROOT_DIR)
  foreach(ARG ${Lab_Aperture_src})
    list(REMOVE_ITEM Aperture_src ${ARG})
    list(APPEND Aperture_src "${LAB_ROOT_DIR}/src/${ARG}")
  endforeach()
endif()

add_library(Aperture ${Aperture_src})
target_link_libraries(Aperture ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt)

if (CMAKE_CUDA_COMPILER)
  set(cuAperture_src
    "cuda/data/fields_utils.cu"
    "cuda/data/particles_dev.cu"
    "cuda/data/photons_dev.cu"
    "cuda/data/particles_1d.cu"
    "cuda/data/photons_1d.cu"
    "cuda/data/fields_dev.cu"
    "cuda/data/array.cu"
    "cuda/data/cu_multi_array.cu"
    "cuda/grids/grid_log_sph_dev.cpp"
    "cuda/grids/grid_1dgr_dev.cu"
    "cuda/core/field_solver_log_sph.cu"
    "cuda/core/field_solver_1dgr.cu"
    "cuda/core/sim_environment_dev.cpp"
    "cuda/core/cu_sim_data.cu"
    "cuda/core/cu_sim_data1d.cu"
    "cuda/core/additional_diagnostics.cu"
    "cuda/core/ptc_updater_dev.cu"
    "cuda/core/ptc_updater_logsph.cu"
    "cuda/core/ptc_updater_1dgr.cu"
    "cuda/core/finite_diff.cu"
    # "algorithms/ptc_pusher_constE.cu"
    # "algorithms/ptc_pusher_beadonwire.cu"
    # "algorithms/ptc_pusher_geodesic.cpp"
    "cuda/utils/cu_data_exporter.cpp"
    "cuda/constant_mem.cu"
    "cuda/kernels.cu"
    "cuda/radiation/rt_pulsar.cu"
    )

  if(LAB_ROOT_DIR)
    foreach(ARG ${Lab_cuAperture_src})
      list(REMOVE_ITEM cuAperture_src ${ARG})
      list(APPEND cuAperture_src "${LAB_ROOT_DIR}/src/${ARG}")
    endforeach()
  endif()

  add_cuda_target(cuAperture ${cuAperture_src})
  target_link_libraries(cuAperture ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt ${CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES} Aperture)
endif()

if(LAB_ROOT_DIR)
  message("${Yellow}Using shadowed main.cpp from lab ${LAB_NAME}${ColorReset}")
  add_executable(aperture "${LAB_ROOT_DIR}/src/main.cpp")
  target_link_libraries(aperture Aperture)
else()
  message("${Yellow}Using default main.cpp${ColorReset}")
  add_executable(aperture "main.cpp")
  target_link_libraries(aperture Aperture)
endif()
if (CMAKE_CUDA_COMPILER)
  target_link_libraries(aperture cuAperture)
endif()

add_executable(aperture1D "main_1d.cpp")
target_link_libraries(aperture1D Aperture)

# if (CUDA_FOUND)
#   target_link_libraries(aperture Pulsar_dev)
# endif()

if(LAB_ROOT_DIR)
  set_target_properties(aperture PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LAB_ROOT_DIR})

  if (shadows)
    add_custom_target( LABREVIEW ALL
      ${CMAKE_COMMAND} -E cmake_echo_color --yellow "The following are shadowed by LAB ${LAB_NAME}")

    #TODO: display shadowed headers also
    foreach(ARG ${Lab_Aperture_src})
      add_custom_command(TARGET LABREVIEW POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --yellow "    ${ARG}")
    endforeach()
    # if(Lab_User_Control_Found)
    #   add_custom_command(TARGET LABREVIEW POST_BUILD
    #     COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --yellow "    UserControl.cpp")
    # endif()

    add_dependencies( LABREVIEW aperture )
  endif()
else()
  add_custom_target( LABREVIEW ALL
    ${CMAKE_COMMAND} -E cmake_echo_color --yellow "NO lab is invoked")
endif()
