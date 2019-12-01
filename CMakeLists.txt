cmake_minimum_required(VERSION 3.8)
# include(ExternalProject)
project (Aperture LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

message ("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")

option(build_tests "Build the test suite." ON)
option(use_cuda "Include the GPU part of the code" OFF)

# enable colored cmake message.
# Copied from https://stackoverflow.com/questions/18968979/how-to-get-colorized-output-with-cmake
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(Yellow      "${Esc}[33m")
endif()

if(POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()

# Set CXX flags
set(CMAKE_CXX_STANDARD 14)
set(CXX_EXTRA_FLAGS "-DBOOST_ALL_NO_LIB=1 -DBOOST_NO_CXX14_CONSTEXPR")
# message(${CMAKE_CXX_COMPILER})
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "-xHost -no-prec-div ${CXX_EXTRA_FLAGS}")
else()
  # set(CXX_EXTRA_FLAGS "-Wno-misleading-indentation -Wno-unused-parameter")
  set(CXX_EXTRA_FLAGS "${CXX_EXTRA_FLAGS} -Wno-unused-parameter")
  set(CMAKE_CXX_FLAGS_RELEASE "-mtune=native ${CXX_EXTRA_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CXX_EXTRA_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CXX_EXTRA_FLAGS} -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS "-pthread -Wall -Wextra -fPIC")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(LAB_ROOT_DIR)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LAB_ROOT_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${LAB_ROOT_DIR})
else()
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
endif()
set(HDF5_USE_STATIC_LIBRARIES)

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
# set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(OpenMP)
if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(FMT_INCLUDE_DIR deps/fmt)
set(CUDALE_INCLUDE_DIR deps/CudaLE/include)
set(JSON_INCLUDE_DIR deps/json/include)
set(CXX_OPTS_INCLUDE_DIR deps/cxxopts/include)
set(CPPTOML_INCLUDE_DIR deps/cpptoml/include)
set(VISIT_STRUCT_INCLUDE_DIR deps/visit_struct/include)
set(HIGHFIVE_INCLUDE_DIR deps/highfive/include)
set(CATCH_INCLUDE_DIR deps/catch/single_include)
set(VECTOR_INCLUDE_DIR deps/vectorclass)

include_directories(${MPI_INCLUDE_PATH} ${FMT_INCLUDE_DIR}
  ${CUDALE_INCLUDE_DIR} ${JSON_INCLUDE_DIR} ${CXX_OPTS_INCLUDE_DIR} ${CPPTOML_INCLUDE_DIR}
  ${CATCH_INCLUDE_DIR} ${HIGHFIVE_INCLUDE_DIR}
  ${VISIT_STRUCT_INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${VECTOR_INCLUDE_DIR} include src)

######### Check for cuda support
if (use_cuda)
  message("Checking cuda support")
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message("Enabling Cuda, ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS}")
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CUDA_INCLUDE_DIRS})
    set(CMAKE_CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr \
      -lcublas -lcurand -Xcompiler -fPIC -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored \
      -Xcudafe --diag_suppress=code_is_unreachable -ccbin=${CMAKE_CXX_COMPILER}")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
    set(CMAKE_CUDA_FLAGS_RELEASE "-lineinfo -O3 -DNDEBUG")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    # set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    message(${CMAKE_CXX_COMPILER})
    macro(add_cuda_target cuda_target)
      add_library(${cuda_target} ${${cuda_target}_src})
      # target_compile_options(${cuda_target} PRIVATE "-arch=compute_61;-code=sm_61;-lineinfo")
    endmacro()
    macro(add_cuda_executable cuda_target)
      add_executable(${cuda_target} ${${cuda_target}_src})
      # target_compile_options(${cuda_target} PRIVATE "-arch=compute_61;-code=sm_61;-lineinfo")
    endmacro()
  endif()
endif()

if(use_double)
  add_definitions(-DUSE_DOUBLE)
endif()
add_subdirectory(deps/fmt)
add_subdirectory(src)
# add_subdirectory(tools)

if (build_tests)
  # Enable testing
  enable_testing(true)

  add_subdirectory(tests)
endif()
