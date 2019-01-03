add_custom_target(check)

# add_executable(test_hdf5 "hdf5.cpp")
# target_link_libraries(test_hdf5 ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt Aperture)

# add_executable(test_f "test_f.cpp")
# target_link_libraries(test_f ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt Aperture)

# add_executable(test_free_path "test_free_path.cpp")
# target_link_libraries(test_free_path ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt Aperture)

set(test_fd_src
  "test.cpp" "test_finite_diff.cpp")
add_executable(test_fd ${test_fd_src})
target_link_libraries(test_fd
  # PUBLIC AperCuda
  PUBLIC Aperture)
add_dependencies(check test_fd)

set(tests_src
  "test.cpp" "test_AD.cpp"
  "test_config.cpp"
  "test_env.cpp"
  "test_stagger.cpp"
  "test_particles.cpp"
  "test_multi_array.cpp"
  "test_fields.cpp"
  "test_field_solver.cpp"
  "test_avx.cpp"
  "test_ptc_update.cpp"
  # "test_finite_diff.cpp"
  # "test_ffe.cpp" "test_ffe2d.cpp"
  # "test_IC_PL.cpp"
  # "test_pair_creation.cpp"
  )
add_executable(tests ${tests_src})
target_link_libraries(tests
  # PUBLIC AperCuda
  PUBLIC Aperture)

add_custom_target(run_tests
  COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests"
  DEPENDS tests
  WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
add_dependencies(check run_tests)

# if(CUDA_FOUND)
if (CMAKE_CUDA_COMPILER)
  set(test_cuda_src
    "test.cu" "test_multiarray.cu")
  # "test_pair_creation.cu")
  # "test_cuda.cu" "test_particles.cu")
  add_cuda_executable(test_cuda)
  set_target_properties(test_cuda PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(test_cuda
    PUBLIC Aperture_cuda
    PUBLIC Aperture)
  add_custom_target(run_cuda_tests
    COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_cuda"
    DEPENDS test_cuda
    WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
  add_dependencies(check run_cuda_tests)
endif()

add_executable(test_parse "test_parse.cpp")
target_link_libraries(test_parse Aperture)
