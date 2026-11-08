// device_report.cu — the machine constants of one GPU, as results/<device>/device.txt.
//
// Every device-dependent number the analysis needs is measured or queried here, so that moving
// the campaign to another GPU does not mean editing constants in a Python script: the roofs drawn
// by analysis/make_i16_results.py are read from this file.
//
// Two bandwidths are reported and they are not interchangeable. triad_bw_gbs is a read-read-write
// stream; read_bw_gbs is a pure read stream, and that is the roof SpMV has to be quoted against,
// because SpMV writes only O(rows) and reads everything else. The analysis uses read_bw_gbs.
//
// Usage:   build/device_report [device id] > results/<device>/device.txt
//
// The advertised capacity and memory kind are not queryable (the query rounds the capacity down and
// says nothing of the kind), so no memory= line is written; add one by hand, e.g. memory=64~GB HBM2e,
// for the setup paragraph to name it. Without it the analysis derives the capacity from the query.

#include "common.cuh"
#include "roofline.cuh"

#include <cstdio>
#include <cstdlib>
#include <string>

// Marketing architecture name of a compute capability, for the prose of the setup paragraph.
static const char* architecture_name(int compute_capability) {
  switch (compute_capability / 10) {
    case 6:  return "Pascal";
    case 7:  return (compute_capability == 75) ? "Turing" : "Volta";
    case 8:  return (compute_capability == 80) ? "Ampere" : "Ampere/Ada";
    case 9:  return "Hopper";
    case 10: case 12: return "Blackwell";
    default: return "unknown";
  }
}

int main(int argc, char** argv) {
  const int  device_id = (argc > 1) ? std::atoi(argv[1]) : 0;
  DeviceInfo device;
  PeakFlops  peak;
  double     read_bandwidth  = 0.0;
  double     triad_bandwidth = 0.0;
  int        runtime_version = 0;
  int        driver_version  = 0;
  int        sparse_major    = 0;
  int        sparse_minor    = 0;
  int        sparse_patch    = 0;

  device = query_device(device_id);
  peak   = peak_flops(device);
  scale_peak_by_clock(peak, device.clock_ghz);
  read_bandwidth  = measure_read_bandwidth(device);
  triad_bandwidth = measure_stream_bandwidth(device);
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUSPARSE_CHECK(cusparseGetProperty(MAJOR_VERSION, &sparse_major));
  CUSPARSE_CHECK(cusparseGetProperty(MINOR_VERSION, &sparse_minor));
  CUSPARSE_CHECK(cusparseGetProperty(PATCH_LEVEL,   &sparse_patch));

  std::printf("# written by build/device_report; the memory= line is not queryable and is added by hand\n");
  std::printf("name=%s\n", device.name.c_str());
  std::printf("arch=%s\n", architecture_name(device.cc()));
  std::printf("cc=%d.%d\n", device.cc_major, device.cc_minor);
  std::printf("sm=%d\n", device.sm_count);
  std::printf("clock_ghz=%g\n", device.clock_ghz);
  std::printf("global_mem_gib=%g\n", device.global_mem / 1073741824.0);
  std::printf("l2_mib=%g\n", device.l2_bytes / 1048576.0);
  std::printf("threads_per_sm=%d\n", device.threads_per_sm);
  std::printf("theoretical_bw_gbs=%g\n", device.peak_bw_gbs);
  std::printf("measured_bw_gbs=%g\n", triad_bandwidth);   // kept: earlier name of triad_bw_gbs
  std::printf("triad_bw_gbs=%g\n", triad_bandwidth);
  std::printf("read_bw_gbs=%g\n", read_bandwidth);
  if (peak.known) {
    std::printf("peak_fp64_gflops=%g\n", peak.fp64_gflops);
    std::printf("peak_fp32_gflops=%g\n", peak.fp32_gflops);
    std::printf("peak_fp16_gflops=%g\n", peak.fp16_gflops);
    std::printf("peak_tc_fp16_gflops=%g\n", peak.tc_fp16_gflops);
  } else {
    std::fprintf(stderr, "compute capability %d unknown to peak_flops(): add it to include/roofline.cuh\n",
                 device.cc());
  }
  std::printf("fp64_rate_divisor=%d\n", device.fp64_rate_divisor());
  std::printf("cuda=%d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
  std::printf("driver=%d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
  std::printf("cusparse=%d.%d.%d\n", sparse_major, sparse_minor, sparse_patch);
  return 0;
}
