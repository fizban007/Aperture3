# add_executable(test_hdf5 "hdf5.cpp")
# target_link_libraries(test_hdf5 ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt Aperture)

add_executable(test_f "test_f.cpp")
target_link_libraries(test_f ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt Aperture)

add_executable(test_free_path "test_free_path.cpp")
target_link_libraries(test_free_path ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${Silo_LIBRARIES} ${HDF5_LIBRARIES} fmt Aperture)
