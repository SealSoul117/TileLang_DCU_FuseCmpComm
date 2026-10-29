cmake_minimum_required(VERSION 3.21)
project(tilelang_dushmem_persistent_16 LANGUAGES CXX HIP)

if(NOT DEFINED ENV{DTKROOT})
    message(FATAL_ERROR "DTKROOT must point at the DTK installation")
endif()

# DTK 26.04 installs DUSHMEMConfig.cmake below lib/dushmem, but that file
# derives an incorrect installation prefix and consequently cannot locate the
# headers. Import the installed libraries directly until it is fixed upstream.
add_library(dushmem_host SHARED IMPORTED)
set_target_properties(dushmem_host PROPERTIES
    IMPORTED_LOCATION "$ENV{DTKROOT}/lib/dushmem/libdushmem_host.so"
    INTERFACE_INCLUDE_DIRECTORIES "$ENV{DTKROOT}/include;$ENV{DTKROOT}/include/dushmem")

add_library(dushmem_device STATIC IMPORTED)
set_target_properties(dushmem_device PROPERTIES
    IMPORTED_LOCATION "$ENV{DTKROOT}/lib/dushmem/libdushmem_device.a"
    INTERFACE_INCLUDE_DIRECTORIES "$ENV{DTKROOT}/include;$ENV{DTKROOT}/include/dushmem")

add_executable(persistent_gemm_allreduce_16 persistent_gemm_allreduce_16.hip)
add_executable(serial_gemm_dushmem_allreduce serial_gemm_dushmem_allreduce.hip)

foreach(target persistent_gemm_allreduce_16 serial_gemm_dushmem_allreduce)
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../..//src")
    target_link_libraries(${target} PRIVATE dushmem_host dushmem_device dl)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3;-fgpu-rdc;-mcode-object-version=4>)
    target_link_options(${target} PRIVATE -fgpu-rdc -mcode-object-version=4)
    target_compile_definitions(${target} PRIVATE
        HIP_ENABLE_WARP_SYNC_BUILTINS __STDC_LIMIT_MACROS __STDC_CONSTANT_MACROS)
    set_target_properties(${target} PROPERTIES
        HIP_ARCHITECTURES gfx936
        HIP_SEPARABLE_COMPILATION ON
        BUILD_RPATH "$ENV{DTKROOT}/lib/dushmem;$ENV{DTKROOT}/hip/lib;$ENV{DTKROOT}/lib")
endforeach()

# Build a separate ROC_SHMEM variant when a locally installed/extracted
# ROC_SHMEM prefix is provided. Keep the DUSHMEM binaries above available for
# side-by-side correctness and performance comparisons.
if(DEFINED ENV{ROCSHMEM_ROOT})
    find_package(MPI REQUIRED COMPONENTS CXX)
    find_package(Threads REQUIRED)
    find_library(NUMA_LIBRARY NAMES numa REQUIRED)

    add_library(rocshmem STATIC IMPORTED)
    set_target_properties(rocshmem PROPERTIES
        IMPORTED_LOCATION "$ENV{ROCSHMEM_ROOT}/lib/librocshmem.a"
        INTERFACE_INCLUDE_DIRECTORIES "$ENV{ROCSHMEM_ROOT}/include")

    add_executable(persistent_gemm_allreduce_rocshmem
        persistent_gemm_allreduce_16.hip)
    target_include_directories(persistent_gemm_allreduce_rocshmem PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../..//src")
    target_link_libraries(persistent_gemm_allreduce_rocshmem PRIVATE
        rocshmem MPI::MPI_CXX Threads::Threads ${NUMA_LIBRARY}
        "$ENV{DTKROOT}/lib/libhsa-runtime64.so" dl)
    target_compile_features(persistent_gemm_allreduce_rocshmem PRIVATE cxx_std_17)
    target_compile_options(persistent_gemm_allreduce_rocshmem PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3;-fgpu-rdc>)
    target_link_options(persistent_gemm_allreduce_rocshmem PRIVATE
        -fgpu-rdc)
    target_compile_definitions(persistent_gemm_allreduce_rocshmem PRIVATE
        TILELANG_USE_ROCSHMEM HIP_ENABLE_WARP_SYNC_BUILTINS
        __STDC_LIMIT_MACROS __STDC_CONSTANT_MACROS)
    set_target_properties(persistent_gemm_allreduce_rocshmem PROPERTIES
        HIP_ARCHITECTURES gfx936
        HIP_SEPARABLE_COMPILATION ON
        BUILD_RPATH "$ENV{DTKROOT}/hip/lib;$ENV{DTKROOT}/lib;$ENV{MPI_HOME}/lib")
endif()
