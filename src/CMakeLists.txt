if(LAB_ROOT_DIR)
  include("${LAB_ROOT_DIR}/src/CMakeLists.txt")
  include_directories(BEFORE "${LAB_ROOT_DIR}/include")
  include_directories(BEFORE "${LAB_ROOT_DIR}/src")
  set(shadows "${LAB_SHADOWS}")
  if (shadows)
    message("${Yellow}Invoked LAB ${LAB_NAME} from ${LAB_ROOT_DIR}, with following shadowed${ColorReset}")
  endif()
  set(Lab_Aperture_src ${shadows})
  foreach(ARG ${LAB_SHADOWS})
      message("${Yellow}    ${ARG}${ColorReset}")
  endforeach()
else()
  message("${Yellow}No lab is invoked.${ColorReset}")
endif()

set(Aperture_src
  "utils/logger.cpp"
  "utils/timer.cpp"
  "utils/mpi_helper.cpp"
  # "utils/mpi_comm.cpp"
  "utils/memory.cpp"
  "utils/filesystem.cpp"
  "utils/data_exporter.cpp"
  "core/grid.cpp"
  "commandline_args.cpp"
  "config_file.cpp"
  "grids/grid_log_sph.cpp"
  "grids/grid_1dgr.cpp"
  "grids/grid_1dgap.cpp"
  "data/detail/particle_data_impl.cpp"
  )
if (use_cuda AND CMAKE_CUDA_COMPILER)
  list(APPEND Aperture_src
    "cuda/kernels.cu"
    "cuda/constant_mem.cu"
    # "cuda/utils/pitchptr.cu"
    "cuda/core/array.cu"
    "cuda/core/multi_array.cu"
    "cuda/core/fields.cu"
    "cuda/core/particles.cu"
    "cuda/core/photons.cu"
    "cuda/sim_data.cu"
    "cuda/sim_environment.cpp"
    "cuda/grids/grid_log_sph_ptrs.cu"
    "cuda/algorithms/field_solver_logsph.cu"
    "cuda/algorithms/field_solver_1dgr.cu"
    "cuda/algorithms/field_solver_1dgap.cu"
    # "cuda/algorithms/field_solver_ffe.cu"
    "cuda/algorithms/ffe_solver_logsph.cu"
    "cuda/algorithms/ptc_updater.cu"
    "cuda/algorithms/ptc_updater_logsph.cu"
    "cuda/algorithms/ptc_updater_cart.cu"
    "cuda/algorithms/ptc_updater_1dgr.cu"
    "cuda/algorithms/ptc_updater_1dgap.cu"
    "cuda/radiation/radiative_transfer.cu"
    "cuda/radiation/rt_ic.cu"
    "cuda/core/finite_diff.cu"
    )
else()
  list(APPEND Aperture_src
    "core/array.cpp"
    "core/multi_array.cpp"
    "core/fields.cpp"
    "core/particles.cpp"
    "core/photons.cpp"
    "sim_data.cpp"
    "sim_environment.cpp"
    # "grids/grid_log_sph.cpp"
    # "data/photons.cpp"
    # "utils/hdf_exporter.cpp"
    # "utils/data_exporter.cpp"
    # "utils/exporter.cpp"
    # "sim_environment.cpp"
    # "sim_data.cpp"
    # "core/field_solver_default.cpp"
    # "core/field_solver_1d.cpp"
    # "core/ptc_updater_default.cpp"
    # "core/ptc_updater_1d.cpp"
    # "radiation/rt_1d.cpp"
    )
endif()

if(LAB_ROOT_DIR)
  foreach(ARG ${Lab_Aperture_src})
    list(REMOVE_ITEM Aperture_src ${ARG})
    list(APPEND Aperture_src "${LAB_ROOT_DIR}/src/${ARG}")
  endforeach()
endif()

add_library(Aperture ${Aperture_src})
target_link_libraries(Aperture ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES}
  ${HDF5_LIBRARIES} fmt ${CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES})
# else()
#   add_library(Aperture ${Aperture_src})
#   target_link_libraries(Aperture ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES}
#     ${HDF5_LIBRARIES} fmt)
# endif()

add_executable(aperture)
if(LAB_ROOT_DIR)
  message("${Yellow}Using shadowed main.cpp from lab ${LAB_NAME}${ColorReset}")
  target_sources(aperture PUBLIC "${LAB_ROOT_DIR}/src/main.cpp")
else()
  message("${Yellow}Using default main.cpp${ColorReset}")
  target_sources(aperture PUBLIC "main.cpp")
endif()
target_link_libraries(aperture Aperture)

# if (LAB_ROOT_DIR)
# else()
#   add_executable(aperture1D "main_1d.cpp")
#   target_link_libraries(aperture1D Aperture)
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
