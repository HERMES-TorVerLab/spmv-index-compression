#pragma once
// value_types.cuh — the value precisions of the benchmark: FP16 accumulates in FP32, the others in
// their own precision, so CSR-i32 and every narrow-index layout sum in the same order and type.

#include <cuda_fp16.h>

// Built with SPMV_HALF_ACCUMULATE, FP16 accumulates in FP16 as well, a control experiment on what
// the widening to FP32 costs; the paper's campaign uses the default build.
template <class T> struct AccumulatorOf         { using type = float;  };
template <>        struct AccumulatorOf<double> { using type = double; };
#ifdef SPMV_HALF_ACCUMULATE
template <>        struct AccumulatorOf<__half> { using type = __half; };
#endif

__host__ __device__ __forceinline__ double widen_value(double value) { return value; }
__host__ __device__ __forceinline__ float  widen_value(float value)  { return value; }
__host__ __device__ __forceinline__ float  widen_value(__half value) { return __half2float(value); }

__host__ __device__ __forceinline__ void narrow_value(double accumulated, double* destination) { *destination = accumulated; }
__host__ __device__ __forceinline__ void narrow_value(float accumulated,  float* destination)  { *destination = accumulated; }
__host__ __device__ __forceinline__ void narrow_value(float accumulated,  __half* destination) { *destination = __float2half(accumulated); }
__host__ __device__ __forceinline__ void narrow_value(__half accumulated, __half* destination) { *destination = accumulated; }

// sum += value * x_value in the accumulator type: widened and fused in FP32 or FP64, or one FP16 FMA
// when FP16 accumulates in FP16.
template <class A, class T>
__device__ __forceinline__ void accumulate_product(A& sum, T value, T x_value) {
  sum += widen_value(value) * widen_value(x_value);
}
#ifdef SPMV_HALF_ACCUMULATE
template <>
__device__ __forceinline__ void accumulate_product<__half, __half>(__half& sum, __half value, __half x_value) {
  sum = __hfma(value, x_value, sum);
}
#endif

template <class T> struct PrecisionName;
template <> struct PrecisionName<double> { static constexpr const char* text = "fp64"; };
template <> struct PrecisionName<float>  { static constexpr const char* text = "fp32"; };
template <> struct PrecisionName<__half> { static constexpr const char* text = "fp16"; };
