cmake_minimum_required(VERSION 3.20)
project(spmv_index_width LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set the compute capability of the target GPU explicitly when building for another machine,
# e.g. cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80 (A100); the default builds for the local GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(CUDAToolkit REQUIRED)

# GPU benchmark: CSR with 32/16/8-bit and row-relative column indices, delta-encoded layouts,
# optional RCM reordering, and cuSPARSE as reference.
add_executable(bench_csr_width src/bench_csr_width.cu)
target_include_directories(bench_csr_width PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_csr_width PRIVATE CUDA::cudart CUDA::cusparse CUDA::cublas)

# Control experiment: the same benchmark at FP16 only, accumulating in FP16 instead of FP32.
add_executable(bench_csr_width_fp16acc src/bench_csr_width.cu)
target_include_directories(bench_csr_width_fp16acc PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_csr_width_fp16acc PRIVATE SPMV_HALF_ACCUMULATE)
target_link_libraries(bench_csr_width_fp16acc PRIVATE CUDA::cudart CUDA::cusparse CUDA::cublas)

# Control experiment: the same benchmark with the CSR-r16 loop unrolled at FP16 as CSR-i32 is.
add_executable(bench_csr_width_r16unroll src/bench_csr_width.cu)
target_include_directories(bench_csr_width_r16unroll PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(bench_csr_width_r16unroll PRIVATE SPMV_R16_UNROLL16)
target_link_libraries(bench_csr_width_r16unroll PRIVATE CUDA::cudart CUDA::cusparse CUDA::cublas)

# Control experiments on the bytes per load of CSR-r16: lane-major offsets with 16-bit loads, the same
# stream with one 32-bit load per two nonzeros, and offsets stored in 32 bits.
foreach(control lane pair wide)
  string(TOUPPER "${control}" control_upper)
  add_executable(bench_csr_width_r16${control} src/bench_csr_width.cu)
  target_include_directories(bench_csr_width_r16${control} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  target_compile_definitions(bench_csr_width_r16${control} PRIVATE SPMV_R16_${control_upper})
  target_link_libraries(bench_csr_width_r16${control} PRIVATE CUDA::cudart CUDA::cusparse CUDA::cublas)
endforeach()

# Machine constants of the GPU (peaks, measured bandwidths, versions) as results/<device>/device.txt.
add_executable(device_report src/device_report.cu)
target_include_directories(device_report PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(device_report PRIVATE CUDA::cudart CUDA::cusparse CUDA::cublas)

# Host only: structural metrics and byte-model ceilings of one matrix, optionally after RCM.
add_executable(characterize_matrix src/characterize_matrix.cpp)
target_include_directories(characterize_matrix PRIVATE ${CMAKE_SOURCE_DIR}/include)

# On a cluster the toolkit comes from a module, and the module environment of a batch job is not
# always the one the build had: a job that loads an older CUDA finds an older libcusparse, and the
# binary then dies at the first call to a symbol that version does not export, after queuing. Baking
# the library directory of the toolkit that compiled the code into the binary removes that class of
# failure entirely, because the loader no longer depends on LD_LIBRARY_PATH being right.
foreach(target bench_csr_width bench_csr_width_fp16acc bench_csr_width_r16unroll bench_csr_width_r16lane
               bench_csr_width_r16pair bench_csr_width_r16wide device_report)
  set_target_properties(${target} PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")
endforeach()

message(STATUS "CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDAToolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA libraries     : ${CUDAToolkit_LIBRARY_DIR} (baked into the RPATH)")
