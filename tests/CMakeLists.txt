add_custom_target(check)

# add_executable(test_hdf5 "hdf5.cpp")
# target_link_libraries(test_hdf5 ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt Aperture)

# add_executable(test_f "test_f.cpp")
# target_link_libraries(test_f ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt Aperture)

# add_executable(test_free_path "test_free_path.cpp")
# target_link_libraries(test_free_path ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt Aperture)

# add_executable(tests ${tests_src})
add_executable(tests "test.cpp")
target_sources(tests PRIVATE
  # "test_stagger.cpp"
  "test_multi_array.cpp"
  "test_flags.cpp"
  # "test_fields.cpp"
  # "test_field_solver.cpp"
  # "test_avx.cpp"
  # "test_ptc_update.cpp"
)
target_link_libraries(tests
  # PUBLIC AperCuda
  PUBLIC Aperture)

add_custom_target(run_tests
  COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests"
  DEPENDS tests
  WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
add_dependencies(check run_tests)

if (use_cuda AND CMAKE_CUDA_COMPILER)
  add_executable(test_fd "test.cpp")
  target_sources(test_fd PRIVATE "test_finite_diff.cpp")
  target_link_libraries(test_fd
    PUBLIC Aperture)
  add_dependencies(check test_fd)

  # "test_pair_creation.cu")
  # "test_cuda.cu" "test_particles.cu")
  add_executable(test_cuda "test.cu")
  target_sources(test_cuda PUBLIC
    # "test_multiarray.cu"
    # "test_particles.cpp"
    # "test_env.cpp"
    # "test_domain.cpp"
    "test_pitchptr.cu"
    # "test_config.cpp"
    # "test_ic.cu"
    )
  # set_target_properties(test_cuda PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(test_cuda
    PUBLIC Aperture)
  add_custom_target(run_cuda_tests
    COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_cuda"
    DEPENDS test_cuda
    WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  add_dependencies(check run_cuda_tests)
endif()

# add_executable(test_parse "test_parse.cpp")
# target_link_libraries(test_parse Aperture)

# add_executable(test_IC "test_IC.cpp")
# target_link_libraries(test_IC Aperture)

# add_executable(print_data_size "data.cpp")
# target_link_libraries(print_data_size Aperture)
