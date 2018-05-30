add_custom_target(check)

# add_executable(test_hdf5 "hdf5.cpp")
# target_link_libraries(test_hdf5 ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt Aperture)

add_executable(test_f "test_f.cpp")
target_link_libraries(test_f ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt Aperture)

add_executable(test_free_path "test_free_path.cpp")
target_link_libraries(test_free_path ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt Aperture)

set(tests_src "test.cpp" "test_AD.cpp")
add_executable(tests EXCLUDE_FROM_ALL ${tests_src})
target_link_libraries(tests Aperture)

add_custom_target(run_tests
  COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests"
  DEPENDS tests
  WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
add_dependencies(check run_tests)

# if(CUDA_FOUND)
set(test_cuda_src "test_cuda.cu")
add_cuda_executable(test_cuda)
add_custom_target(run_cuda_tests
  COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_cuda"
  DEPENDS test_cuda
  WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
add_dependencies(check run_cuda_tests)
# endif()

