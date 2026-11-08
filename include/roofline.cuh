#pragma once
// roofline.cuh — the two ceilings a SpMV number has to be quoted against.
//
// 1. MEASURED streaming bandwidth (a triad over a buffer several times L2).
//    Theoretical clock*bus bandwidth is never reachable; quoting SpMV as a
//    fraction of it understates every kernel by 15-25%. Both are reported.
// 2. Peak FMA throughput on the CUDA cores, from the arithmetic-instruction
//    table. For SpMV this ceiling is the one that is *never* approached, and
//    saying so quantitatively is the point of the exercise.

#include "common.cuh"
#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// Peak arithmetic throughput (CUDA cores, FMA = 2 flop)
// ---------------------------------------------------------------------------

struct PeakFlops {
  int    fp32_cores_per_sm = 0;
  int    fp64_cores_per_sm = 0;
  double clock_ghz         = 0.0;
  double fp32_gflops       = 0.0;
  double fp64_gflops       = 0.0;
  double fp16_gflops       = 0.0;   // CUDA cores, fp16_per_fp32 times the fp32 rate
  double tc_fp16_gflops    = 0.0;   // tensor cores, fp16 in / fp32 acc
  bool   known             = false;
};

// Cores/SM from the CUDA C Programming Guide "Arithmetic Instructions" table.
// The fp64 column is the one that breaks the "compute capability" abstraction:
// 7.5 and 8.6/8.9 are 2 fp64 units/SM, 7.0/8.0/9.0 are 32/32/64.
inline PeakFlops peak_flops(const DeviceInfo& device) {
  PeakFlops peak;
  int       tensor_cores_per_sm   = 0;
  int       flops_per_tensor_core = 0;   // fp16*fp16+fp32 flops per clock per tensor core
  // FP16 on the CUDA cores against FP32, which changes with the architecture: twice on Volta, Turing and
  // Hopper, four times on the A100 (78 against 19.5 TFLOP/s in its data sheet), once on GA10x and Ada.
  double    fp16_per_fp32         = 2.0;

  peak.clock_ghz = 0.0;                  // filled by caller from the clock attribute of the device
  switch (device.cc()) {
    case 60:            peak.fp32_cores_per_sm = 64;  peak.fp64_cores_per_sm = 32; break;
    case 61: case 62:   peak.fp32_cores_per_sm = 128; peak.fp64_cores_per_sm = 4;  break;
    case 70: case 72:   peak.fp32_cores_per_sm = 64;  peak.fp64_cores_per_sm = 32;
                        tensor_cores_per_sm = 8;  flops_per_tensor_core = 128; break;
    case 75:            peak.fp32_cores_per_sm = 64;  peak.fp64_cores_per_sm = 2;
                        tensor_cores_per_sm = 8;  flops_per_tensor_core = 128; break;  // Turing, 2nd-gen TC
    case 80:            peak.fp32_cores_per_sm = 64;  peak.fp64_cores_per_sm = 32;
                        tensor_cores_per_sm = 4;  flops_per_tensor_core = 512; fp16_per_fp32 = 4.0; break;
    case 86: case 87: case 89:
                        peak.fp32_cores_per_sm = 128; peak.fp64_cores_per_sm = 2;
                        tensor_cores_per_sm = 4;  flops_per_tensor_core = 512; fp16_per_fp32 = 1.0; break;
    case 90:            peak.fp32_cores_per_sm = 128; peak.fp64_cores_per_sm = 64;
                        tensor_cores_per_sm = 4;  flops_per_tensor_core = 1024; break;
    default: return peak;  // unknown -- leave known=false, print "n/a"
  }
  peak.known          = true;
  peak.fp32_gflops    = 2.0 * peak.fp32_cores_per_sm * device.sm_count;   // * GHz, applied later
  peak.fp64_gflops    = 2.0 * peak.fp64_cores_per_sm * device.sm_count;
  peak.fp16_gflops    = fp16_per_fp32 * peak.fp32_gflops;
  peak.tc_fp16_gflops = (double)tensor_cores_per_sm * flops_per_tensor_core * device.sm_count;
  return peak;
}

inline void scale_peak_by_clock(PeakFlops& peak, double clock_ghz) {
  peak.clock_ghz       = clock_ghz;
  peak.fp32_gflops    *= clock_ghz;
  peak.fp64_gflops    *= clock_ghz;
  peak.fp16_gflops    *= clock_ghz;
  peak.tc_fp16_gflops *= clock_ghz;
}

// ---------------------------------------------------------------------------
// Measured streaming bandwidth
// ---------------------------------------------------------------------------

__global__ static void k_triad(double* __restrict__ destination, const double* __restrict__ first,
                               const double* __restrict__ second, double scale, size_t length) {
  size_t index  = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;

  for (; index < length; index += stride) destination[index] = first[index] + scale * second[index];
}

// Pure-read bandwidth. This, not the triad, is the correct roof for GEMV and
// SpMV: those kernels write only O(rows) and read everything else, so they
// legitimately exceed a triad number that includes a full write stream. On the
// Quadro RTX 5000 the triad gives 380-384 GB/s and this gives 407.9 GB/s, and
// the dense GEMV kernel reaches 404.8 -- i.e. 99.2% of the read roof, which is
// only interpretable against the right ceiling.
__global__ static void k_read_bw(const double* __restrict__ source, size_t length, double* sink) {
  size_t index       = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride      = (size_t)gridDim.x * blockDim.x;
  double accumulator = 0.0;

  for (; index < length; index += stride) accumulator += source[index];
  if (accumulator == 1234.5678) sink[0] = accumulator;   // never true; defeats dead-code removal
}

inline double measure_read_bandwidth(const DeviceInfo& device, int reps = 20) {
  const int  block_size   = 256;
  size_t     free_bytes   = 0;
  size_t     total_bytes  = 0;
  size_t     length       = 0;
  int        grid_size    = 0;
  double*    source       = nullptr;
  double*    sink         = nullptr;
  EventTimer timer;

  (void)device;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  length    = (size_t)(0.5 * free_bytes) / sizeof(double);
  grid_size = (int)std::min<size_t>((length + block_size - 1) / block_size, 65535 * 4);
  CUDA_CHECK(cudaMalloc(&source, length * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&sink, sizeof(double)));
  CUDA_CHECK(cudaMemset(source, 0, length * sizeof(double)));
  for (int warmup = 0; warmup < 5; ++warmup) k_read_bw<<<grid_size, block_size>>>(source, length, sink);
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int rep = 0; rep < reps; ++rep) {
    timer.start();
    k_read_bw<<<grid_size, block_size>>>(source, length, sink);
    timer.stop();
  }
  cudaFree(source); cudaFree(sink);
  return (double)length * sizeof(double) / (timer.median_ms() * 1e-3) / 1e9;
}

// Returns GB/s. Buffer sized to comfortably exceed L2 so this is a DRAM number.
inline double measure_stream_bandwidth(const DeviceInfo& device, int reps = 20) {
  const int  block_size   = 256;
  size_t     free_bytes   = 0;
  size_t     total_bytes  = 0;
  size_t     length       = 0;
  int        grid_size    = 0;
  double*    destination  = nullptr;
  double*    first        = nullptr;
  double*    second       = nullptr;
  double     bytes_moved  = 0.0;
  EventTimer timer;

  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  length = (size_t)(0.35 * free_bytes) / (3 * sizeof(double));
  length = std::min<size_t>(length, 64ull << 20);                          // cap at 512 MiB/array
  length = std::max<size_t>(length, (size_t)device.l2_bytes * 8 / sizeof(double));
  if ((size_t)3 * length * sizeof(double) > (size_t)(0.8 * free_bytes))
    length = (size_t)(0.8 * free_bytes) / (3 * sizeof(double));
  grid_size = (int)std::min<size_t>((length + block_size - 1) / block_size, 65535 * 4);

  CUDA_CHECK(cudaMalloc(&destination, length * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&first,       length * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&second,      length * sizeof(double)));
  CUDA_CHECK(cudaMemset(first,  1, length * sizeof(double)));
  CUDA_CHECK(cudaMemset(second, 1, length * sizeof(double)));

  for (int warmup = 0; warmup < 5; ++warmup)
    k_triad<<<grid_size, block_size>>>(destination, first, second, 2.0, length);
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int rep = 0; rep < reps; ++rep) {
    timer.start();
    k_triad<<<grid_size, block_size>>>(destination, first, second, 2.0, length);
    timer.stop();
  }
  CUDA_CHECK(cudaGetLastError());

  bytes_moved = 3.0 * length * sizeof(double);   // 2 read + 1 write (write-allocate ignored)
  cudaFree(destination); cudaFree(first); cudaFree(second);
  return bytes_moved / (timer.median_ms() * 1e-3) / 1e9;
}
