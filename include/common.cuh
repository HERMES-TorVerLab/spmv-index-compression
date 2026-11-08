#pragma once
// common.cuh — error handling, CUDA-event timing, device introspection.

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "[CUDA] %s:%d: %s (%s)\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(_e), #call);                             \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t _s = (call);                                                \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
      std::fprintf(stderr, "[cuBLAS] %s:%d: status %d (%s)\n", __FILE__,       \
                   __LINE__, (int)_s, #call);                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Hard failure.
#define CUSPARSE_CHECK(call)                                                   \
  do {                                                                         \
    cusparseStatus_t _s = (call);                                              \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                       \
      std::fprintf(stderr, "[cuSPARSE] %s:%d: %s (%s)\n", __FILE__, __LINE__,  \
                   cusparseGetErrorString(_s), #call);                         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Soft failure: used when probing whether a precision combination is supported
// on this device / toolkit. Returns the status instead of aborting. This is the
// mechanism behind the "support matrix by compute capability" table.
#define CUSPARSE_TRY(call, status_out)                                         \
  do {                                                                         \
    (status_out) = (call);                                                     \
  } while (0)

inline bool is_unsupported(cusparseStatus_t s) {
  return s == CUSPARSE_STATUS_NOT_SUPPORTED ||
         s == CUSPARSE_STATUS_INVALID_VALUE ||
         s == CUSPARSE_STATUS_ARCH_MISMATCH;
}

// ---------------------------------------------------------------------------
// Device info
// ---------------------------------------------------------------------------

struct DeviceInfo {
  int         id          = 0;
  std::string name;
  int         cc_major    = 0;
  int         cc_minor    = 0;
  size_t      global_mem  = 0;
  double      clock_ghz     = 0.0;  // SM boost clock, the factor of every peak flop rate
  int         mem_clock_khz = 0;
  int         mem_bus_bits  = 0;
  double      peak_bw_gbs   = 0.0;  // theoretical, from clock * bus width
  int         sm_count      = 0;
  size_t      l2_bytes      = 0;    // decisive for SpMV: the working set relative to L2 decides cache residency
  int         threads_per_sm = 0;   // resident threads a multiprocessor holds, which with the threads
                                    // per row fixes how many rows of the matrix are in flight at once

  int cc() const { return cc_major * 10 + cc_minor; }

  // Capability gates that matter for this study.
  bool has_fp16_native() const { return cc() >= 53; }
  bool has_bf16()        const { return cc() >= 80; }
  bool has_tf32()        const { return cc() >= 80; }
  bool has_fp8()         const { return cc() >= 89; }

  // FP64:FP32 throughput ratio, from the CUDA C Programming Guide arithmetic
  // instruction tables. This is the number that makes "compute capability" a
  // misleading label: CC 8.0 (A100) is 1:2 with FP64 tensor cores, CC 8.6/8.9
  // (consumer & pro Ada/Ampere) is 1:64. Same major version, 32x difference.
  int fp64_rate_divisor() const {
    switch (cc()) {
      case 60: case 70: case 80: case 90: return 2;   // GP100/V100/A100/H100
      case 61: case 62:                   return 32;  // Pascal consumer
      case 75:                            return 32;  // Turing
      case 86: case 87: case 89:          return 64;  // Ampere/Ada consumer & pro
      case 100: case 101: case 120:       return 64;  // Blackwell (verify!)
      default:                            return 0;   // unknown -- check the guide
    }
  }
};

// The two clocks are read as attributes: CUDA 13 removed clockRate and memoryClockRate from
// cudaDeviceProp, and cudaDeviceGetAttribute returns the same kHz values on CUDA 12 as well.
inline DeviceInfo query_device(int dev = 0) {
  cudaDeviceProp p{};
  DeviceInfo     d;
  int            clock_khz        = 0;
  int            memory_clock_khz = 0;

  CUDA_CHECK(cudaSetDevice(dev));
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, dev));

  d.id            = dev;
  d.name          = p.name;
  d.cc_major      = p.major;
  d.cc_minor      = p.minor;
  d.global_mem    = p.totalGlobalMem;
  d.clock_ghz     = clock_khz / 1e6;
  d.mem_clock_khz = memory_clock_khz;
  d.mem_bus_bits  = p.memoryBusWidth;
  d.sm_count       = p.multiProcessorCount;
  d.l2_bytes       = (size_t)p.l2CacheSize;
  d.threads_per_sm = p.maxThreadsPerMultiProcessor;
  // DDR -> factor 2. Result in GB/s (10^9).
  d.peak_bw_gbs =
      2.0 * (double)memory_clock_khz * 1e3 * (double)p.memoryBusWidth / 8.0 / 1e9;
  return d;
}

inline void print_device(const DeviceInfo& d, std::FILE* out = stdout) {
  std::fprintf(out, "device      : %s (id %d)\n", d.name.c_str(), d.id);
  std::fprintf(out, "compute cap : %d.%d\n", d.cc_major, d.cc_minor);
  std::fprintf(out, "SMs         : %d\n", d.sm_count);
  std::fprintf(out, "global mem  : %.2f GiB\n", d.global_mem / 1073741824.0);
  std::fprintf(out, "L2 cache    : %.1f MiB\n", d.l2_bytes / 1048576.0);
  std::fprintf(out, "resident    : %d threads per SM\n", d.threads_per_sm);
  std::fprintf(out, "peak BW     : %.1f GB/s (theoretical)\n", d.peak_bw_gbs);
  std::fprintf(out, "fp64:fp32   : 1:%d (hardware rate)\n", d.fp64_rate_divisor());
  std::fprintf(out, "fp16 native : %s\n", d.has_fp16_native() ? "yes" : "no");
  std::fprintf(out, "bf16        : %s\n", d.has_bf16() ? "yes" : "no");
  std::fprintf(out, "tf32        : %s\n", d.has_tf32() ? "yes" : "no");
  std::fprintf(out, "fp8         : %s\n", d.has_fp8() ? "yes" : "no");
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

// CUDA-event timer. Records a sample per call; caller extracts min/median/max.
// Median (not mean) is the headline: laptop GPUs throttle and the mean hides it.
class EventTimer {
public:
  EventTimer() {
    CUDA_CHECK(cudaEventCreate(&beg_));
    CUDA_CHECK(cudaEventCreate(&end_));
  }
  ~EventTimer() {
    cudaEventDestroy(beg_);
    cudaEventDestroy(end_);
  }
  EventTimer(const EventTimer&)            = delete;
  EventTimer& operator=(const EventTimer&) = delete;

  void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(beg_, s)); }

  // Returns elapsed ms and stores the sample.
  float stop(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(end_, s));
    CUDA_CHECK(cudaEventSynchronize(end_));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg_, end_));
    samples_.push_back(ms);
    return ms;
  }

  void clear() { samples_.clear(); }

  size_t count() const { return samples_.size(); }

  double min_ms() const {
    if (samples_.empty()) return 0.0;
    return *std::min_element(samples_.begin(), samples_.end());
  }
  double max_ms() const {
    if (samples_.empty()) return 0.0;
    return *std::max_element(samples_.begin(), samples_.end());
  }
  double median_ms() const {
    if (samples_.empty()) return 0.0;
    std::vector<float> v = samples_;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }
  double mean_ms() const {
    if (samples_.empty()) return 0.0;
    double s = 0.0;
    for (float x : samples_) s += x;
    return s / samples_.size();
  }

private:
  cudaEvent_t        beg_{}, end_{};
  std::vector<float> samples_;
};
