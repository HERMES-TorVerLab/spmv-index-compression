// bench_csr_width.cu — SpMV with narrow column indices at FP64, FP32 and FP16, against CSR-i32 and
// cuSPARSE, one vector of TPR threads per row.
//
// Column indices come in three modes, the row pointer staying the int32 CSR one:
//   absolute     classic CSR with b-byte columns, admissible only if n <= 2^(8b) (i32, i16, i8).
//   row-base     col = row_base[row] + offset[k], one int32 base per row (its first column) and
//                b-byte offsets, admissible only if every row spans fewer than 2^(8b) columns (r16, r8).
//   lane-packed  CSR-lp: the delta of an entry is taken from the entry TPR positions earlier in its
//                row (from the row base for the first TPR), bit-packed at the width of the row's widest
//                delta and laid out lane by lane, so every lane decodes its own chain. The stream depends
//                on TPR, so it is rebuilt for every candidate.
// Inadmissible configurations are reported and not run.
//
// For every matrix the program prints the memory occupancy of every configuration, then measures
// each at TPR in {2, 4, 8, 16, 32}. Every result is checked against CSR-i32 of the same precision at
// the same TPR, which sums in the same order, so the difference must be exactly zero.
//
// cuSPARSE reference: cusparseSpMV with CSR_ALG1 and CSR_ALG2 on the same int32 arrays. Its summation
// order differs, so it is compared by max |dy| against CSR-i32, not by bit identity.

#include "common.cuh"
#include "csr.hpp"
#include "rcm.hpp"
#include "value_types.cuh"
#include "roofline.cuh"

#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

// Control builds of CSR-r16, each changing how the offsets reach the kernel and nothing else:
//   SPMV_R16_LANE   offsets laid out lane by lane, one 16-bit load per nonzero, which isolates the layout
//   SPMV_R16_PAIR   the same stream, one 32-bit load per two nonzeros of the lane: the same bytes with
//                   fewer loads, so each load carries more of them
//   SPMV_R16_WIDE   offsets stored in 32 bits, which gives the loads of CSR-i32 their bytes back and
//                   keeps the base and the addition of CSR-r16
#if defined(SPMV_R16_LANE) || defined(SPMV_R16_PAIR)
#define SPMV_R16_LANE_MAJOR
#endif

enum class ColumnMode { absolute, row_base, lane_packed };

static const char* column_mode_name(ColumnMode mode) {
  switch (mode) {
    case ColumnMode::lane_packed: return "lane-packed";
    case ColumnMode::row_base:    return "row-base";
    default:                      return "absolute";
  }
}

struct BenchArgs {
  std::vector<std::string> matrix_paths;
  std::string              csv_path      = "results/csr_width.csv";
  int                      reps          = 30;
  int                      warmup        = 10;
  std::string              ordering      = "original";   // original | rcm (P A P^T, permutation cached next to the .mtx)
  double                   min_sample_ms = 0.0;   // > 0: each timing sample runs enough consecutive products to last this long
  std::string              tpr_csv_path;          // non-empty: also write the time of every layout at every threads-per-row
  std::string              precision     = "all";        // all | fp64 | fp32 | fp16: the value precisions to measure
};

// One threads-per-row candidate of one layout. The model compares two layouts at the same assignment
// of threads to rows, while the sweep keeps the fastest of each, so the time at every candidate is
// written out and the comparison can be made at whichever threads per row the analysis chooses.
struct ThreadSample {
  int    threads_per_row = 0;
  int    batch           = 1;
  double time_ms         = 0.0;
  double bytes_total     = 0.0;
  double max_abs_diff    = 0.0;
};

// Upper bound on the products aggregated into one timing sample.
static constexpr int MAX_BATCH = 1 << 20;

template <class T> struct CudaValueType;
template <> struct CudaValueType<double> { static constexpr cudaDataType type = CUDA_R_64F; };
template <> struct CudaValueType<float>  { static constexpr cudaDataType type = CUDA_R_32F; };
template <> struct CudaValueType<__half> { static constexpr cudaDataType type = CUDA_R_16F; };

static const char* cusparse_algorithm_name(cusparseSpMVAlg_t algorithm) {
  return algorithm == CUSPARSE_SPMV_CSR_ALG1 ? "cusparse_csr_alg1" : "cusparse_csr_alg2";
}

struct Configuration {
  int         value_bytes  = 8;
  int         column_bytes = 4;
  ColumnMode  column_mode  = ColumnMode::absolute;
  int         fixed_tpr    = 0;   // 0: sweep threads per row; otherwise the one CSR-lp was built for
};

struct MemoryBreakdown {
  double values = 0, columns = 0, column_bases = 0, row_pointer = 0, vectors = 0, total = 0;
};

struct ResultRow {
  std::string matrix, precision, encoding, column_mode, status = "ok", kernel = "custom", ordering = "original";
  int     value_bytes = 0, column_bytes = 0, length_bytes = 0, rows = 0, cols = 0, best_tpr = 0, batch = 1;
  int64_t nnz = 0;
  MemoryBreakdown memory;
  double  time_ms = 0, gflops = 0, bw_gbs = 0, epsilon = 0;
  double  speedup_measured = 0, speedup_predicted = 0, max_abs_diff = 0, workspace_bytes = 0, max_rel_diff = 0;

  static std::string header() {
    return "matrix,prec,row_encoding,value_bytes,column_bytes,length_bytes,status,rows,cols,nnz,"
           "bytes_values,bytes_columns,bytes_row_pointer,bytes_vectors,bytes_total,bytes_per_nnz,"
           "best_tpr,t_ms,gflops,bw_gbs,epsilon,speedup_meas,speedup_pred,max_abs_diff,column_mode,bytes_column_bases,"
           "kernel,bytes_workspace,max_rel_diff,ordering,batch";
  }
  std::string to_csv() const {
    char line[1024];

    std::snprintf(line, sizeof(line),
      "%s,%s,%s,%d,%d,%d,%s,%d,%d,%lld,%.0f,%.0f,%.0f,%.0f,%.0f,%.4f,%d,%.6f,%.3f,%.2f,%.4f,%.4f,%.4f,%.3e,%s,%.0f,%s,%.0f,%.3e,%s,%d",
      matrix.c_str(), precision.c_str(), encoding.c_str(), value_bytes, column_bytes, length_bytes,
      status.c_str(), rows, cols, (long long)nnz, memory.values, memory.columns, memory.row_pointer,
      memory.vectors, memory.total, memory.total / (double)nnz, best_tpr, time_ms, gflops, bw_gbs,
      epsilon, speedup_measured, speedup_predicted, max_abs_diff, column_mode.c_str(), memory.column_bases,
      kernel.c_str(), workspace_bytes, max_rel_diff, ordering.c_str(), batch);
    return line;
  }
};

template <class T>
struct DeviceMatrix {
  int       rows          = 0;
  int       cols          = 0;
  int64_t   nnz           = 0;
  T*        values        = nullptr;
  T*        x             = nullptr;
  T*        y             = nullptr;
  int*      row_ptr       = nullptr;
  int*      col_ind_32    = nullptr;
  uint16_t* col_ind_16    = nullptr;
  uint8_t*  col_ind_8     = nullptr;
  int*      row_base      = nullptr;
  uint16_t* col_rel_16    = nullptr;
  uint8_t*  col_rel_8     = nullptr;
  uint32_t* col_rel_32    = nullptr;
  uint16_t* col_rel_lane[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};   // by log2(TPR)
  int*      lane_row_word    = nullptr;
  uint8_t*  lane_row_width   = nullptr;
  uint32_t* lane_words       = nullptr;
};

// ---------------------------------------------------------------------------
// Widths and byte model
// ---------------------------------------------------------------------------

static int required_bits(uint64_t largest_value) {
  int bits = 0;

  while (bits < 64 && (largest_value >> bits) != 0) ++bits;
  return std::max(bits, 1);
}

struct MatrixFacts {
  int     widest_row_span = 0;   // max over rows of (last column - first column)
  int64_t lane_packed_words[6] = {0, 0, 0, 0, 0, 0};   // 32-bit words of the CSR-lp stream, indexed by log2(TPR)
};

// Bits of a row's packed deltas: 0 when every delta is 0 (a row with at most one entry).
static int delta_width_bits(int64_t widest_delta) {
  return widest_delta == 0 ? 0 : required_bits((uint64_t)widest_delta);
}

// log2 of a threads-per-row value in {1, 2, ..., 32}.
static int tpr_slot(int threads_per_row) {
  int slot = 0;

  while ((1 << slot) < threads_per_row && slot < 5) ++slot;
  return slot;
}

// Delta of an entry from the entry TPR positions earlier in its row, or from the row's first
// column for the first TPR entries. With TPR = 1 this is the ordinary delta (0 for the first entry).
static int64_t lane_delta(const CsrMatrix& matrix, int row_begin, int entry, int threads_per_row) {
  return entry - row_begin < threads_per_row
             ? (int64_t)matrix.col_ind[entry] - matrix.col_ind[row_begin]
             : (int64_t)matrix.col_ind[entry] - matrix.col_ind[entry - threads_per_row];
}

// Bit position of an entry inside its row's packed stream, in lane-major order.
//
// The deltas of a row are laid out one lane at a time rather than one entry at a time: lane L owns
// entries L, L+T, L+2T, ... and its deltas sit next to each other. That is what lets a thread hold
// the stream in a register and refill it once every 32 bits instead of reading two words for every
// nonzero, which is where the earlier element-major layout spent four global loads per nonzero
// against the three of CSR-i32.
//
// Lane L owns ceil((length - L) / T) entries, so the lanes before it own L*quotient + min(L, remainder)
// of them, and that count is where its run starts. The total is unchanged, length * width bits, so
// the packed size and the byte model are exactly what they were.
static int64_t lane_major_bit(int offset, int length, int threads_per_row, int width) {
  const int quotient  = length / threads_per_row;
  const int remainder = length - quotient * threads_per_row;
  const int lane      = offset % threads_per_row;
  const int step      = offset / threads_per_row;

  return (int64_t)width * ((int64_t)lane * quotient + (lane < remainder ? lane : remainder) + step);
}

static int64_t packed_word_count(const CsrMatrix& matrix, int threads_per_row) {
  int64_t word_count   = 0;
  int64_t widest_delta = 0;
  int     row_begin    = 0;
  int     row_end      = 0;

  for (int row = 0; row < matrix.rows; ++row) {
    row_begin    = matrix.row_ptr[row];
    row_end      = matrix.row_ptr[row + 1];
    widest_delta = 0;
    for (int entry = row_begin; entry < row_end; ++entry)
      widest_delta = std::max(widest_delta, lane_delta(matrix, row_begin, entry, threads_per_row));
    word_count += ((int64_t)(row_end - row_begin) * delta_width_bits(widest_delta) + 31) / 32;
  }
  return word_count + 2;
}

static MatrixFacts facts_of(const CsrMatrix& matrix) {
  MatrixFacts facts;
  int         row_begin = 0;
  int         row_end   = 0;

  for (int row = 0; row < matrix.rows; ++row) {
    row_begin = matrix.row_ptr[row];
    row_end   = matrix.row_ptr[row + 1];
    if (row_end > row_begin)
      facts.widest_row_span = std::max(facts.widest_row_span, matrix.col_ind[row_end - 1] - matrix.col_ind[row_begin]);
  }
  for (int slot = 1; slot <= 5; ++slot) facts.lane_packed_words[slot] = packed_word_count(matrix, 1 << slot);
  return facts;
}

static bool column_width_admissible(const CsrMatrix& matrix, int column_bytes) {
  return column_bytes >= 4 || (uint64_t)matrix.cols <= ((uint64_t)1 << (8 * column_bytes));
}

static bool row_base_admissible(const MatrixFacts& facts, int column_bytes) {
  return column_bytes < 4 && (uint64_t)facts.widest_row_span < ((uint64_t)1 << (8 * column_bytes));
}

static bool configuration_admissible(const CsrMatrix& matrix, const MatrixFacts& facts,
                                     const Configuration& configuration) {
  if (configuration.column_mode == ColumnMode::absolute)
    return column_width_admissible(matrix, configuration.column_bytes);
  if (configuration.column_mode == ColumnMode::row_base)
    return row_base_admissible(facts, configuration.column_bytes);
  return facts.lane_packed_words[tpr_slot(configuration.fixed_tpr > 0 ? configuration.fixed_tpr : 8)] <= INT32_MAX;
}

// Bytes of every stream one SpMV touches. Row metadata beyond the row pointer is charged to
// column_bases: the row base (4 bytes per row) for row-base, plus the word start (4 bytes) and the bit
// width (1 byte) per row for lane-packed.
static MemoryBreakdown memory_of(const CsrMatrix& matrix, const MatrixFacts& facts, const Configuration& configuration) {
  const int       lane_slot = tpr_slot(configuration.fixed_tpr > 0 ? configuration.fixed_tpr : 8);
  const double    nnz       = (double)matrix.nnz();
  MemoryBreakdown memory;

  memory.values      = nnz * configuration.value_bytes;
  memory.columns     = nnz * configuration.column_bytes;
  if (configuration.column_mode == ColumnMode::lane_packed) memory.columns = 4.0 * facts.lane_packed_words[lane_slot];
  memory.row_pointer = (matrix.rows + 1.0) * 4;
  memory.vectors     = (double)(matrix.cols + matrix.rows) * configuration.value_bytes;
  memory.column_bases = 0.0;
  if (configuration.column_mode == ColumnMode::row_base)    memory.column_bases = 4.0 * matrix.rows;
#ifdef SPMV_R16_WIDE
  if (configuration.column_mode == ColumnMode::row_base && configuration.column_bytes == 2) memory.columns = nnz * 4;
#endif
  if (configuration.column_mode == ColumnMode::lane_packed) memory.column_bases = 9.0 * matrix.rows;
  memory.total       = memory.values + memory.columns + memory.column_bases + memory.row_pointer + memory.vectors;
  return memory;
}

// Short label: i32/i16/i8 absolute, r16/r8 row base, lp lane-packed.
static std::string column_label(const Configuration& configuration) {
  switch (configuration.column_mode) {
    case ColumnMode::row_base:    return "r" + std::to_string(configuration.column_bytes * 8);
    case ColumnMode::lane_packed: return "lp";
    default:                      return "i" + std::to_string(configuration.column_bytes * 8);
  }
}

// ---------------------------------------------------------------------------
// Kernels (warp-per-row, uniform trip count)
// ---------------------------------------------------------------------------

template <class T, class I, int TPR>
__global__ void k_csr_cumulative(int rows, const int* __restrict__ row_ptr, const I* __restrict__ col_ind,
                                 const T* __restrict__ values, const T* __restrict__ x, T* __restrict__ y) {
  using Accumulator = typename AccumulatorOf<T>::type;
  // 64-bit thread arithmetic: rows * TPR exceeds 2^31 on the largest matrices
  // (72M rows at TPR = 32), and an int thread id then wraps to a negative row.
  const int64_t thread_id       = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int     lane            = threadIdx.x & (TPR - 1);
  const int64_t vector_id       = thread_id / TPR;
  const int64_t vector_count    = ((int64_t)gridDim.x * blockDim.x) / TPR;
  const int64_t iteration_count = (rows + vector_count - 1) / vector_count;
  int64_t     row             = 0;
  int         row_end         = 0;
  Accumulator sum             = Accumulator(0);

  for (int64_t iteration = 0; iteration < iteration_count; ++iteration) {
    row = vector_id + iteration * vector_count;
    sum = Accumulator(0);
    if (row < rows) {
      row_end = row_ptr[row + 1];
      for (int entry = row_ptr[row] + lane; entry < row_end; entry += TPR)
        accumulate_product(sum, values[entry], x[(int)col_ind[entry]]);
    }
#pragma unroll
    for (int offset = TPR / 2; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset, TPR);
    if (lane == 0 && row < rows) narrow_value(sum, &y[row]);
  }
}

// Row-relative columns: same shape as k_csr_cumulative, plus one base read per row.
template <class T, class I, int TPR>
__global__ void k_csr_row_relative(int rows, const int* __restrict__ row_ptr, const int* __restrict__ row_base,
                               const I* __restrict__ offsets, const T* __restrict__ values,
                               const T* __restrict__ x, T* __restrict__ y) {
  using Accumulator = typename AccumulatorOf<T>::type;
  const int64_t thread_id       = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int     lane            = threadIdx.x & (TPR - 1);
  const int64_t vector_id       = thread_id / TPR;
  const int64_t vector_count    = ((int64_t)gridDim.x * blockDim.x) / TPR;
  const int64_t iteration_count = (rows + vector_count - 1) / vector_count;
  int64_t     row             = 0;
  int         row_end         = 0;
  int         base            = 0;
  Accumulator sum             = Accumulator(0);

  for (int64_t iteration = 0; iteration < iteration_count; ++iteration) {
    row = vector_id + iteration * vector_count;
    sum = Accumulator(0);
    if (row < rows) {
      row_end = row_ptr[row + 1];
      base    = row_base[row];
#ifdef SPMV_R16_UNROLL16
      // Control build: at FP16 nvcc unrolls this loop by 4 instead of the 16 it chooses for CSR-i32
      // and for the other precisions, which leaves a quarter of the independent loads in flight per
      // thread. This build forces 16 at FP16 so that the two layouts differ only in the index bytes.
      if constexpr (std::is_same<T, __half>::value) {
#pragma unroll 16
        for (int entry = row_ptr[row] + lane; entry < row_end; entry += TPR)
          accumulate_product(sum, values[entry], x[base + (int)offsets[entry]]);
      } else {
        for (int entry = row_ptr[row] + lane; entry < row_end; entry += TPR)
          accumulate_product(sum, values[entry], x[base + (int)offsets[entry]]);
      }
#else
      for (int entry = row_ptr[row] + lane; entry < row_end; entry += TPR)
        accumulate_product(sum, values[entry], x[base + (int)offsets[entry]]);
#endif
    }
#pragma unroll
    for (int offset = TPR / 2; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset, TPR);
    if (lane == 0 && row < rows) narrow_value(sum, &y[row]);
  }
}

#ifdef SPMV_R16_LANE_MAJOR
// Row-relative offsets laid out lane by lane in the same 2 * nnz bytes. The row keeps its positions
// in the stream, so no metadata is added and every lane finds where its run starts from the length of
// the row. A word of SPMV_R16_PAIR can straddle two lanes, which reads no byte outside the stream.
template <class T, int TPR>
__global__ void k_csr_row_relative_lane(int rows, const int* __restrict__ row_ptr, const int* __restrict__ row_base,
                                        const uint16_t* __restrict__ offsets, const T* __restrict__ values,
                                        const T* __restrict__ x, T* __restrict__ y) {
  using Accumulator = typename AccumulatorOf<T>::type;
  const int64_t thread_id       = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int     lane            = threadIdx.x & (TPR - 1);
  const int64_t vector_id       = thread_id / TPR;
  const int64_t vector_count    = ((int64_t)gridDim.x * blockDim.x) / TPR;
  const int64_t iteration_count = (rows + vector_count - 1) / vector_count;
  int64_t     row             = 0;
  int         row_begin       = 0;
  int         length          = 0;
  int         quotient        = 0;
  int         remainder       = 0;
  int         mine            = 0;
  int         entry           = 0;
  int         position        = 0;
  int         base            = 0;
  Accumulator sum             = Accumulator(0);
#ifdef SPMV_R16_PAIR
  const uint32_t* words  = reinterpret_cast<const uint32_t*>(offsets);
  uint32_t        word   = 0;
  int             offset = 0;
#endif

  for (int64_t iteration = 0; iteration < iteration_count; ++iteration) {
    row = vector_id + iteration * vector_count;
    sum = Accumulator(0);
    if (row < rows) {
      row_begin = row_ptr[row];
      length    = row_ptr[row + 1] - row_begin;
      base      = row_base[row];
      quotient  = length / TPR;
      remainder = length - quotient * TPR;
      mine      = quotient + (lane < remainder ? 1 : 0);
      entry     = row_begin + lane;
      position  = row_begin + lane * quotient + (lane < remainder ? lane : remainder);
#ifdef SPMV_R16_PAIR
      if (mine > 0) word = words[position >> 1];
      for (int taken = 0; taken < mine; ++taken) {
        offset = (int)((position & 1) ? (word >> 16) : (word & 0xffffu));
        accumulate_product(sum, values[entry], x[base + offset]);
        entry += TPR;
        ++position;
        if ((position & 1) == 0 && taken + 1 < mine) word = words[position >> 1];
      }
#else
      for (int taken = 0; taken < mine; ++taken) {
        accumulate_product(sum, values[entry], x[base + (int)offsets[position]]);
        entry += TPR;
        ++position;
      }
#endif
    }
#pragma unroll
    for (int shift = TPR / 2; shift > 0; shift >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, shift, TPR);
    if (lane == 0 && row < rows) narrow_value(sum, &y[row]);
  }
}
#endif

// Lane-major bit-packed delta columns. Lane L walks entries L, L+TPR, L+2*TPR, ... of its row, and
// its deltas sit next to each other in the stream, so the lane holds the stream in a register pair
// and pulls one new word only when fewer than width bits are left. That is one global load per 32
// bits consumed, against the two loads per nonzero an element-major layout forces, and it is the
// shape Tang et al. describe for the same format.
//
// The running column is still carried across the loop, since a delta is relative to the entry TPR
// positions earlier, but it now depends on a register rather than on a load.
//
// Left to itself the compiler gives this kernel 40 registers, which on a device with 2048 threads
// per multiprocessor caps it at three quarters of the warps the other layouts keep resident, and
// the warps it loses are the requests in flight a latency-bound matrix needs. The bound asks for
// sixteen blocks of 128 threads, the block size every launcher here uses, which fits in 32
// registers without spilling on sm_80. On a device with 1024 threads per multiprocessor the bound
// cannot be met and the compiler ignores it.
template <class T, int TPR>
__global__ void __launch_bounds__(128, 16)
    k_csr_lane_packed(int rows, const int* __restrict__ row_ptr, const int* __restrict__ row_base,
                      const int* __restrict__ row_word, const uint8_t* __restrict__ row_width,
                      const uint32_t* __restrict__ words, const T* __restrict__ values,
                      const T* __restrict__ x, T* __restrict__ y) {
  using Accumulator = typename AccumulatorOf<T>::type;
  const int64_t thread_id       = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int     lane            = threadIdx.x & (TPR - 1);
  const int64_t vector_id       = thread_id / TPR;
  const int64_t vector_count    = ((int64_t)gridDim.x * blockDim.x) / TPR;
  const int64_t iteration_count = (rows + vector_count - 1) / vector_count;
  int64_t     row             = 0;
  int         row_begin       = 0;
  int         length          = 0;
  int         quotient        = 0;
  int         remainder       = 0;
  int         mine            = 0;
  int         entry           = 0;
  int         column          = 0;
  int         width           = 0;
  int         available       = 0;
  int64_t     start_bit       = 0;
  int64_t     word            = 0;
  int64_t     next_word       = 0;
  uint64_t    window          = 0;
  uint64_t    mask            = 0;
  Accumulator sum             = Accumulator(0);

  for (int64_t iteration = 0; iteration < iteration_count; ++iteration) {
    row = vector_id + iteration * vector_count;
    sum = Accumulator(0);
    if (row < rows) {
      row_begin = row_ptr[row];
      length    = row_ptr[row + 1] - row_begin;
      width     = row_width[row];
      mask      = width == 0 ? 0ull : (~0ull >> (64 - width));
      quotient  = length / TPR;
      remainder = length - quotient * TPR;
      mine      = quotient + (lane < remainder ? 1 : 0);
      column    = row_base[row];
      entry     = row_begin + lane;
      // where this lane's run starts, and the two words that cover its first bits
      start_bit = (int64_t)width * ((int64_t)lane * quotient + (lane < remainder ? lane : remainder));
      word      = row_word[row] + (start_bit >> 5);
      window    = (((uint64_t)words[word + 1] << 32) | words[word]) >> (start_bit & 31);
      available = 64 - (int)(start_bit & 31);
      next_word = word + 2;
      for (int taken = 0; taken < mine; ++taken) {
        column += (int)(window & mask);
        accumulate_product(sum, values[entry], x[column]);
        window >>= width;
        available -= width;
        entry     += TPR;
        // One word covers at least 32 more bits, and width never exceeds 32, so a single refill is
        // always enough. The two spare words packed_word_count reserves cover the read past the end.
        if (available < width) {
          window    |= (uint64_t)words[next_word] << available;
          available += 32;
          ++next_word;
        }
      }
    }
#pragma unroll
    for (int offset = TPR / 2; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset, TPR);
    if (lane == 0 && row < rows) narrow_value(sum, &y[row]);
  }
}

// ---------------------------------------------------------------------------
// Dispatch: column mode x column type x threads per row
// ---------------------------------------------------------------------------

template <class T, class I, int TPR>
static void launch_absolute_fixed(const DeviceMatrix<T>& device, const I* col_ind) {
  const int block_size = 128;
  const int grid_size  = (int)(((int64_t)device.rows * TPR + block_size - 1) / block_size);

  k_csr_cumulative<T, I, TPR><<<grid_size, block_size>>>(device.rows, device.row_ptr, col_ind,
                                                        device.values, device.x, device.y);
}

template <class T, class I>
static void launch_absolute(int threads_per_row, const DeviceMatrix<T>& device, const I* col_ind) {
  switch (threads_per_row) {
    case  2: launch_absolute_fixed<T, I,  2>(device, col_ind); break;
    case  4: launch_absolute_fixed<T, I,  4>(device, col_ind); break;
    case  8: launch_absolute_fixed<T, I,  8>(device, col_ind); break;
    case 16: launch_absolute_fixed<T, I, 16>(device, col_ind); break;
    default: launch_absolute_fixed<T, I, 32>(device, col_ind); break;
  }
}

template <class T, class I, int TPR>
static void launch_row_base_fixed(const DeviceMatrix<T>& device, const I* offsets) {
  const int block_size = 128;
  const int grid_size  = (int)(((int64_t)device.rows * TPR + block_size - 1) / block_size);

  k_csr_row_relative<T, I, TPR><<<grid_size, block_size>>>(device.rows, device.row_ptr, device.row_base, offsets,
                                                      device.values, device.x, device.y);
}

template <class T, class I>
static void launch_row_base(int threads_per_row, const DeviceMatrix<T>& device, const I* offsets) {
  switch (threads_per_row) {
    case  2: launch_row_base_fixed<T, I,  2>(device, offsets); break;
    case  4: launch_row_base_fixed<T, I,  4>(device, offsets); break;
    case  8: launch_row_base_fixed<T, I,  8>(device, offsets); break;
    case 16: launch_row_base_fixed<T, I, 16>(device, offsets); break;
    default: launch_row_base_fixed<T, I, 32>(device, offsets); break;
  }
}

template <class T, int TPR>
static void launch_lane_packed_fixed(const DeviceMatrix<T>& device) {
  const int block_size = 128;
  const int grid_size  = (int)(((int64_t)device.rows * TPR + block_size - 1) / block_size);

  k_csr_lane_packed<T, TPR><<<grid_size, block_size>>>(device.rows, device.row_ptr, device.row_base,
                                                     device.lane_row_word, device.lane_row_width, device.lane_words,
                                                     device.values, device.x, device.y);
}

#ifdef SPMV_R16_LANE_MAJOR
template <class T, int TPR>
static void launch_row_base_lane_fixed(const DeviceMatrix<T>& device) {
  const int block_size = 128;
  const int grid_size  = (int)(((int64_t)device.rows * TPR + block_size - 1) / block_size);

  k_csr_row_relative_lane<T, TPR><<<grid_size, block_size>>>(device.rows, device.row_ptr, device.row_base,
                                                             device.col_rel_lane[tpr_slot(TPR)], device.values,
                                                             device.x, device.y);
}

template <class T>
static void launch_row_base_lane(int threads_per_row, const DeviceMatrix<T>& device) {
  switch (threads_per_row) {
    case  2: launch_row_base_lane_fixed<T,  2>(device); break;
    case  4: launch_row_base_lane_fixed<T,  4>(device); break;
    case  8: launch_row_base_lane_fixed<T,  8>(device); break;
    case 16: launch_row_base_lane_fixed<T, 16>(device); break;
    default: launch_row_base_lane_fixed<T, 32>(device); break;
  }
}
#endif

template <class T>
static void launch_lane_packed(int threads_per_row, const DeviceMatrix<T>& device) {
  switch (threads_per_row) {
    case  2: launch_lane_packed_fixed<T,  2>(device); break;
    case  4: launch_lane_packed_fixed<T,  4>(device); break;
    case  8: launch_lane_packed_fixed<T,  8>(device); break;
    case 16: launch_lane_packed_fixed<T, 16>(device); break;
    default: launch_lane_packed_fixed<T, 32>(device); break;
  }
}

template <class T>
static void launch(int threads_per_row, const Configuration& configuration, const DeviceMatrix<T>& device) {
  if (configuration.column_mode == ColumnMode::lane_packed) {
    launch_lane_packed<T>(threads_per_row, device);
    return;
  }
  if (configuration.column_mode == ColumnMode::row_base) {
    if (configuration.column_bytes == 1) launch_row_base<T, uint8_t>(threads_per_row, device, device.col_rel_8);
#if defined(SPMV_R16_LANE_MAJOR)
    else                                 launch_row_base_lane<T>(threads_per_row, device);
#elif defined(SPMV_R16_WIDE)
    else                                 launch_row_base<T, uint32_t>(threads_per_row, device, device.col_rel_32);
#else
    else                                 launch_row_base<T, uint16_t>(threads_per_row, device, device.col_rel_16);
#endif
    return;
  }
  switch (configuration.column_bytes) {
    case 1:  launch_absolute<T, uint8_t>(threads_per_row, device, device.col_ind_8);   break;
    case 2:  launch_absolute<T, uint16_t>(threads_per_row, device, device.col_ind_16); break;
    default: launch_absolute<T, int>(threads_per_row, device, device.col_ind_32);      break;
  }
}

// Number of consecutive products per timing sample so that one sample lasts at least
// args.min_sample_ms. A sub-millisecond kernel timed alone is dominated by the timer resolution
// and by launch and synchronisation costs; timing a batch and dividing by its size removes both.
// The batch is grown by 4x until a sample lasts a quarter of the target, then scaled to it.
template <class Run>
static int batch_for(const Run& run_once, const BenchArgs& args) {
  EventTimer timer;
  int        batch     = 1;
  double     sample_ms = 0.0;

  if (args.min_sample_ms <= 0.0) return 1;
  for (;;) {
    timer.start();
    for (int product = 0; product < batch; ++product) run_once();
    sample_ms = timer.stop();
    if (sample_ms >= 0.25 * args.min_sample_ms || batch >= MAX_BATCH) break;
    batch *= 4;
  }
  return (int)std::min<double>(MAX_BATCH, std::max(1.0, std::ceil(batch * args.min_sample_ms / std::max(sample_ms, 1e-3))));
}

// Median time of one product in milliseconds, over args.reps samples of batch products each.
template <class Run>
static double batched_median_ms(const Run& run_once, const BenchArgs& args, int batch) {
  EventTimer timer;

  for (int rep = 0; rep < args.reps; ++rep) {
    timer.start();
    for (int product = 0; product < batch; ++product) run_once();
    timer.stop();
  }
  return timer.median_ms() / batch;
}

template <class T>
static std::vector<double> result_at_fixed_tpr(const Configuration& configuration, const DeviceMatrix<T>& device,
                                               int threads_per_row);

template <class T>
static double best_time(const Configuration& configuration, const DeviceMatrix<T>& device,
                        const BenchArgs& args, int& best_tpr, int& best_batch,
                        std::vector<ThreadSample>* samples = nullptr) {
  const int  swept_tpr[]      = {2, 4, 8, 16, 32};
  const int  fixed_tpr[]      = {configuration.fixed_tpr};
  const bool fixed            = configuration.fixed_tpr > 0;
  const int* candidates       = fixed ? fixed_tpr : swept_tpr;
  const int  candidate_count  = fixed ? 1 : 5;
  double     best_ms          = 1e30;
  double     candidate_ms     = 0.0;
  int        threads_per_row  = 0;
  int        batch            = 1;
  ThreadSample sample;
  const auto launch_once      = [&]() { launch<T>(threads_per_row, configuration, device); };

  if (samples) samples->clear();
  for (int candidate = 0; candidate < candidate_count; ++candidate) {
    threads_per_row = candidates[candidate];
    for (int warmup_rep = 0; warmup_rep < args.warmup; ++warmup_rep) launch_once();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    batch        = batch_for(launch_once, args);
    candidate_ms = batched_median_ms(launch_once, args, batch);
    CUDA_CHECK(cudaGetLastError());
    if (candidate_ms < best_ms) { best_ms = candidate_ms; best_tpr = threads_per_row; best_batch = batch; }
    if (samples) {
      sample                 = ThreadSample{};
      sample.threads_per_row = threads_per_row;
      sample.batch           = batch;
      sample.time_ms         = candidate_ms;
      samples->push_back(sample);
    }
  }
  return best_ms;
}

// Largest |y - y_reference| of a configuration against CSR-i32 of the same precision, both launched at
// the same threads per row so that they accumulate in the same order.
template <class T>
static double max_difference_at(const Configuration& configuration, const DeviceMatrix<T>& device,
                                int threads_per_row) {
  const Configuration reference_configuration = {configuration.value_bytes, 4, ColumnMode::absolute};
  std::vector<double> reference = result_at_fixed_tpr<T>(reference_configuration, device, threads_per_row);
  std::vector<double> candidate = result_at_fixed_tpr<T>(configuration, device, threads_per_row);
  double              largest   = 0.0;

  for (int index = 0; index < device.rows; ++index)
    largest = std::max(largest, std::fabs(candidate[index] - reference[index]));
  return largest;
}

// Time the lane bit-packed layout at every threads-per-row candidate and leave the device holding
// the stream of the one that won.
//
// The other layouts are swept inside best_time for nothing, because changing the threads per row only
// changes the launch. This one cannot be: a delta is taken from the entry that many positions
// earlier, so each candidate needs its own stream built and uploaded. Until now the layout simply
// inherited whatever CSR-i32 had chosen, which measured it at someone else's tuning and is not a
// comparison. Rebuilding costs two passes over the nonzeros per candidate on the host.
template <class T>
static double best_lane_packed_time(const CsrMatrix& matrix, const MatrixFacts& facts,
                                    Configuration configuration, DeviceMatrix<T>& device,
                                    const BenchArgs& args, int& best_tpr, int& best_batch,
                                    std::vector<ThreadSample>* samples = nullptr) {
  const int    candidates[]  = {2, 4, 8, 16, 32};
  double       best_ms       = 1e30;
  double       candidate_ms  = 0.0;
  int          candidate_tpr = 0;
  int          held_tpr      = 0;
  int          batch         = 1;
  int          ignored       = 0;
  ThreadSample sample;

  best_tpr   = 0;
  best_batch = 1;
  if (samples) samples->clear();
  for (int index = 0; index < 5; ++index) {
    candidate_tpr = candidates[index];
    if (facts.lane_packed_words[tpr_slot(candidate_tpr)] > INT32_MAX) continue;
    upload_packed_deltas(matrix, candidate_tpr, facts.lane_packed_words[tpr_slot(candidate_tpr)],
                         device.lane_row_word, device.lane_row_width, device.lane_words);
    held_tpr                = candidate_tpr;
    configuration.fixed_tpr = candidate_tpr;
    candidate_ms            = best_time<T>(configuration, device, args, ignored, batch);
    // The stream of this candidate is on the device only now, so its byte count and its check against
    // CSR-i32 are taken here rather than by the caller.
    if (samples) {
      sample                 = ThreadSample{};
      sample.threads_per_row = candidate_tpr;
      sample.batch           = batch;
      sample.time_ms         = candidate_ms;
      sample.bytes_total     = memory_of(matrix, facts, configuration).total;
      sample.max_abs_diff    = max_difference_at<T>(configuration, device, candidate_tpr);
      samples->push_back(sample);
    }
    if (candidate_ms < best_ms) {
      best_ms    = candidate_ms;
      best_tpr   = candidate_tpr;
      best_batch = batch;
    }
  }
  // The loop leaves the last candidate on the device, which is not always the winner.
  if (best_tpr > 0 && best_tpr != held_tpr)
    upload_packed_deltas(matrix, best_tpr, facts.lane_packed_words[tpr_slot(best_tpr)],
                         device.lane_row_word, device.lane_row_width, device.lane_words);
  return best_ms;
}

template <class T>
static std::vector<double> result_at_fixed_tpr(const Configuration& configuration, const DeviceMatrix<T>& device,
                                               int threads_per_row) {
  std::vector<T>      host_y(device.rows);
  std::vector<double> widened(device.rows);

  launch<T>(threads_per_row, configuration, device);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(host_y.data(), device.y, sizeof(T) * device.rows, cudaMemcpyDeviceToHost));
  for (int row = 0; row < device.rows; ++row) widened[row] = (double)widen_value(host_y[row]);
  return widened;
}

// ---------------------------------------------------------------------------
// Upload / free
// ---------------------------------------------------------------------------

template <class T, class I>
static I* upload_columns(const CsrMatrix& matrix) {
  std::vector<I> host_columns((size_t)matrix.nnz());
  I*             device_columns = nullptr;

  for (int64_t entry = 0; entry < matrix.nnz(); ++entry) host_columns[entry] = (I)matrix.col_ind[entry];
  CUDA_CHECK(cudaMalloc(&device_columns, sizeof(I) * matrix.nnz()));
  CUDA_CHECK(cudaMemcpy(device_columns, host_columns.data(), sizeof(I) * matrix.nnz(), cudaMemcpyHostToDevice));
  return device_columns;
}

// Base of each row: its first column (0 for an empty row).
static int* upload_row_bases(const CsrMatrix& matrix) {
  std::vector<int> host_bases(matrix.rows, 0);
  int*             device_bases = nullptr;

  for (int row = 0; row < matrix.rows; ++row)
    if (matrix.row_ptr[row + 1] > matrix.row_ptr[row]) host_bases[row] = matrix.col_ind[matrix.row_ptr[row]];
  CUDA_CHECK(cudaMalloc(&device_bases, sizeof(int) * matrix.rows));
  CUDA_CHECK(cudaMemcpy(device_bases, host_bases.data(), sizeof(int) * matrix.rows, cudaMemcpyHostToDevice));
  return device_bases;
}

// Column offsets from the row's base, narrowed to I (the caller checks the span fits).
template <class I>
static I* upload_relative_columns(const CsrMatrix& matrix) {
  std::vector<I> host_offsets((size_t)matrix.nnz());
  I*             device_offsets = nullptr;
  int            row_first_column = 0;

  for (int row = 0; row < matrix.rows; ++row) {
    row_first_column = (matrix.row_ptr[row + 1] > matrix.row_ptr[row]) ? matrix.col_ind[matrix.row_ptr[row]] : 0;
    for (int entry = matrix.row_ptr[row]; entry < matrix.row_ptr[row + 1]; ++entry)
      host_offsets[entry] = (I)(matrix.col_ind[entry] - row_first_column);
  }
  CUDA_CHECK(cudaMalloc(&device_offsets, sizeof(I) * matrix.nnz()));
  CUDA_CHECK(cudaMemcpy(device_offsets, host_offsets.data(), sizeof(I) * matrix.nnz(), cudaMemcpyHostToDevice));
  return device_offsets;
}

#ifdef SPMV_R16_LANE_MAJOR
// The 16-bit offsets of every row in lane-major order for one threads per row, in whole 32-bit words so
// that SPMV_R16_PAIR can read the last one.
static uint16_t* upload_lane_major_offsets(const CsrMatrix& matrix, int threads_per_row) {
  std::vector<uint16_t> host_offsets((size_t)((matrix.nnz() + 1) / 2 * 2), 0);
  uint16_t*             device_offsets = nullptr;
  int                   row_begin      = 0;
  int                   length         = 0;
  int                   position       = 0;

  for (int row = 0; row < matrix.rows; ++row) {
    row_begin = matrix.row_ptr[row];
    length    = matrix.row_ptr[row + 1] - row_begin;
    for (int offset = 0; offset < length; ++offset) {
      position = row_begin + (int)lane_major_bit(offset, length, threads_per_row, 1);
      host_offsets[position] = (uint16_t)(matrix.col_ind[row_begin + offset] - matrix.col_ind[row_begin]);
    }
  }
  CUDA_CHECK(cudaMalloc(&device_offsets, sizeof(uint16_t) * host_offsets.size()));
  CUDA_CHECK(cudaMemcpy(device_offsets, host_offsets.data(), sizeof(uint16_t) * host_offsets.size(), cudaMemcpyHostToDevice));
  return device_offsets;
}
#endif

// CSR-lp stream for a given threads per row: per row the word where its deltas start and their bit
// width, the deltas laid out lane by lane (lane_major_bit).
static void build_packed_deltas(const CsrMatrix& matrix, int threads_per_row, int64_t word_count,
                                std::vector<int>& row_word, std::vector<uint8_t>& row_width,
                                std::vector<uint32_t>& words) {
  int64_t  word_cursor  = 0;
  int64_t  widest_delta = 0;
  int64_t  bit          = 0;
  int64_t  word         = 0;
  uint64_t delta        = 0;
  int      shift        = 0;
  int      width        = 0;
  int      row_begin    = 0;
  int      row_end      = 0;

  row_word.assign(matrix.rows, 0);
  row_width.assign(matrix.rows, 0);
  words.assign((size_t)word_count, 0u);
  for (int row = 0; row < matrix.rows; ++row) {
    row_begin    = matrix.row_ptr[row];
    row_end      = matrix.row_ptr[row + 1];
    widest_delta = 0;
    for (int entry = row_begin; entry < row_end; ++entry)
      widest_delta = std::max(widest_delta, lane_delta(matrix, row_begin, entry, threads_per_row));
    width          = delta_width_bits(widest_delta);
    row_word[row]  = (int)word_cursor;
    row_width[row] = (uint8_t)width;
    for (int entry = row_begin; entry < row_end; ++entry) {
      delta = (uint64_t)lane_delta(matrix, row_begin, entry, threads_per_row);
      bit   = lane_major_bit(entry - row_begin, row_end - row_begin, threads_per_row, width);
      word  = word_cursor + (bit >> 5);
      shift = (int)(bit & 31);
      words[word] |= (uint32_t)(delta << shift);
      if (shift + width > 32) words[word + 1] |= (uint32_t)(delta >> (32 - shift));
    }
    word_cursor += ((int64_t)(row_end - row_begin) * width + 31) / 32;
  }
}

static void upload_packed_deltas(const CsrMatrix& matrix, int threads_per_row, int64_t word_count,
                                 int*& device_row_word, uint8_t*& device_row_width, uint32_t*& device_words) {
  std::vector<int>      host_row_word;
  std::vector<uint8_t>  host_row_width;
  std::vector<uint32_t> host_words;

  build_packed_deltas(matrix, threads_per_row, word_count, host_row_word, host_row_width, host_words);
  // The stream depends on the threads per row, so sweeping that parameter calls this more than once
  // per matrix. Release what the previous call left; cudaFree of a null pointer is a no-op, which
  // covers the first one.
  CUDA_CHECK(cudaFree(device_row_word));
  CUDA_CHECK(cudaFree(device_row_width));
  CUDA_CHECK(cudaFree(device_words));
  CUDA_CHECK(cudaMalloc(&device_row_word, sizeof(int) * matrix.rows));
  CUDA_CHECK(cudaMalloc(&device_row_width, sizeof(uint8_t) * matrix.rows));
  CUDA_CHECK(cudaMalloc(&device_words, sizeof(uint32_t) * host_words.size()));
  CUDA_CHECK(cudaMemcpy(device_row_word, host_row_word.data(), sizeof(int) * matrix.rows, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_row_width, host_row_width.data(), sizeof(uint8_t) * matrix.rows, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_words, host_words.data(), sizeof(uint32_t) * host_words.size(), cudaMemcpyHostToDevice));
}

template <class T>
static void upload_matrix(const CsrMatrix& matrix, const MatrixFacts& facts, DeviceMatrix<T>& device) {
  std::vector<T>         host_values((size_t)matrix.nnz());
  std::vector<T>         host_x(matrix.cols);
  std::mt19937           generator(7);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);

  device.rows = matrix.rows; device.cols = matrix.cols; device.nnz = matrix.nnz();
  for (int64_t entry = 0; entry < matrix.nnz(); ++entry) host_values[entry] = (T)(float)matrix.val[entry];
  for (T& value : host_x) value = (T)(float)uniform(generator);

  CUDA_CHECK(cudaMalloc(&device.values, sizeof(T) * device.nnz));
  CUDA_CHECK(cudaMalloc(&device.x, sizeof(T) * device.cols));
  CUDA_CHECK(cudaMalloc(&device.y, sizeof(T) * device.rows));
  CUDA_CHECK(cudaMalloc(&device.row_ptr, sizeof(int) * (device.rows + 1)));
  CUDA_CHECK(cudaMemcpy(device.values, host_values.data(), sizeof(T) * device.nnz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.x, host_x.data(), sizeof(T) * device.cols, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.row_ptr, matrix.row_ptr.data(), sizeof(int) * (device.rows + 1), cudaMemcpyHostToDevice));
  device.col_ind_32 = upload_columns<T, int>(matrix);
  if (column_width_admissible(matrix, 2)) device.col_ind_16 = upload_columns<T, uint16_t>(matrix);
  if (column_width_admissible(matrix, 1)) device.col_ind_8  = upload_columns<T, uint8_t>(matrix);
  // the bases serve row-base and CSR-lp, which is measured on every matrix
  device.row_base = upload_row_bases(matrix);
  if (row_base_admissible(facts, 2)) device.col_rel_16 = upload_relative_columns<uint16_t>(matrix);
#ifdef SPMV_R16_WIDE
  if (row_base_admissible(facts, 2)) device.col_rel_32 = upload_relative_columns<uint32_t>(matrix);
#endif
#ifdef SPMV_R16_LANE_MAJOR
  if (row_base_admissible(facts, 2))
    for (int slot = 1; slot <= 5; ++slot) device.col_rel_lane[slot] = upload_lane_major_offsets(matrix, 1 << slot);
#endif
  if (row_base_admissible(facts, 1)) device.col_rel_8  = upload_relative_columns<uint8_t>(matrix);
}

template <class T>
static void free_matrix(DeviceMatrix<T>& device) {
  cudaFree(device.values); cudaFree(device.x); cudaFree(device.y); cudaFree(device.row_ptr);
  cudaFree(device.col_ind_32); cudaFree(device.col_ind_16); cudaFree(device.col_ind_8);
  cudaFree(device.row_base); cudaFree(device.col_rel_16); cudaFree(device.col_rel_8); cudaFree(device.col_rel_32);
  for (uint16_t* lane_offsets : device.col_rel_lane) cudaFree(lane_offsets);
  cudaFree(device.lane_row_word); cudaFree(device.lane_row_width); cudaFree(device.lane_words);
}

// Device bytes the upload above needs for one precision (all uploaded layouts at once, CSR-lp at its
// widest candidate).
static double device_bytes_needed(const CsrMatrix& matrix, const MatrixFacts& facts, int value_bytes) {
  const double nnz            = (double)matrix.nnz();
  const double rows           = (double)matrix.rows;
  const double widest_lane    = (double)*std::max_element(facts.lane_packed_words + 1, facts.lane_packed_words + 6);
  const double common_bytes   = nnz * value_bytes + (double)(matrix.cols + matrix.rows) * value_bytes
                              + (rows + 1.0) * 4 + nnz * 4;
  const double narrow_bytes   = (column_width_admissible(matrix, 2) ? nnz * 2 : 0.0)
                              + (column_width_admissible(matrix, 1) ? nnz : 0.0)
                              + (row_base_admissible(facts, 1) ? nnz : 0.0);
  const double relative_bytes = (row_base_admissible(facts, 2) ? nnz * 2 : 0.0) + 4.0 * rows;
  const double lane_bytes     = 4.0 * widest_lane + 5.0 * rows;
#if defined(SPMV_R16_LANE_MAJOR)
  const double control_bytes  = row_base_admissible(facts, 2) ? 5.0 * nnz * 2 : 0.0;
#elif defined(SPMV_R16_WIDE)
  const double control_bytes  = row_base_admissible(facts, 2) ? nnz * 4 : 0.0;
#else
  const double control_bytes  = 0.0;
#endif

  return common_bytes + narrow_bytes + relative_bytes + lane_bytes + control_bytes;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

static void append_thread_samples(const std::string& path, const ResultRow& row, const std::string& layout,
                                  const std::vector<ThreadSample>& samples) {
  std::ifstream probe(path);
  const bool    needs_header = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
  std::ofstream output(path, std::ios::app);
  char          line[512];

  if (!output) return;
  if (needs_header) output << "matrix,ordering,prec,layout,tpr,t_ms,batch,bytes_total,nnz,max_abs_diff\n";
  for (const ThreadSample& sample : samples) {
    std::snprintf(line, sizeof(line), "%s,%s,%s,%s,%d,%.6f,%d,%.0f,%lld,%.3e\n", row.matrix.c_str(),
                  row.ordering.c_str(), row.precision.c_str(), layout.c_str(), sample.threads_per_row,
                  sample.time_ms, sample.batch, sample.bytes_total, (long long)row.nnz, sample.max_abs_diff);
    output << line;
  }
}

static void append_row(const std::string& path, const ResultRow& row) {
  std::ifstream probe(path);
  const bool    needs_header = !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
  std::ofstream output(path, std::ios::app);

  if (!output) return;
  if (needs_header) output << ResultRow::header() << "\n";
  output << row.to_csv() << "\n";
}

// Per precision: absolute i32, i16, i8; row-base r16, r8; CSR-lp.
static std::vector<Configuration> configurations_for() {
  const int                  value_widths[]    = {8, 4, 2};
  const int                  column_widths[]   = {4, 2, 1};
  const int                  relative_widths[] = {2, 1};
  std::vector<Configuration> configurations;

  for (int value_bytes : value_widths) {
    for (int column_bytes : column_widths)
      configurations.push_back({value_bytes, column_bytes, ColumnMode::absolute});
    for (int column_bytes : relative_widths)
      configurations.push_back({value_bytes, column_bytes, ColumnMode::row_base});
    configurations.push_back({value_bytes, 0, ColumnMode::lane_packed, 0});
  }
  return configurations;
}

static const char* precision_of(int value_bytes) {
  return value_bytes == 8 ? "fp64" : (value_bytes == 4 ? "fp32" : "fp16");
}

static void print_memory_table(const CsrMatrix& matrix, const MatrixFacts& facts) {
  const double    mebibyte            = 1048576.0;
  MemoryBreakdown memory;
  const char*     inadmissible_reason = nullptr;

  std::printf("  memory occupancy (MiB)  %-4s %-10s %3s %9s %9s %9s %9s %9s %9s %7s %s\n", "prec", "row ptr", "col",
              "values", "columns", "bases", "row ptr", "vectors", "total", "B/nnz", "");
  for (const Configuration& configuration : configurations_for()) {
    if (configuration.column_mode == ColumnMode::lane_packed) continue;
    memory              = memory_of(matrix, facts, configuration);
    inadmissible_reason = configuration.column_mode == ColumnMode::absolute
                              ? "(not admissible: n too large)" : "(not admissible: a row spans too many columns)";
    std::printf("                          %-4s %-10s %-3s %9.1f %9.1f %9.2f %9.2f %9.1f %9.1f %7.3f %s\n",
                precision_of(configuration.value_bytes), "cumulative", column_label(configuration).c_str(),
                memory.values / mebibyte, memory.columns / mebibyte, memory.column_bases / mebibyte,
                memory.row_pointer / mebibyte, memory.vectors / mebibyte, memory.total / mebibyte,
                memory.total / (double)matrix.nnz(),
                configuration_admissible(matrix, facts, configuration) ? "" : inadmissible_reason);
  }
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

// One cuSPARSE algorithm on the device arrays already uploaded (int32 indices).
// The descriptor is created per algorithm: cuSPARSE keeps preprocess state inside
// it and reusing it across algorithms gives silently wrong results.
// FP16 data is computed in FP32, as our kernels accumulate it: cuSPARSE rejects an FP16 compute
// type, and with an FP32 one it takes alpha and beta as float.
template <class T>
static void run_cusparse(cusparseHandle_t sparse_handle, cusparseSpMVAlg_t algorithm, const DeviceMatrix<T>& device,
                         const BenchArgs& args, double read_bandwidth, double baseline_ms,
                         const std::vector<double>& reference_y, double reference_scale, ResultRow row) {
  using Scalar = typename std::conditional<std::is_same<T, __half>::value, float, T>::type;
  const cudaDataType    value_type    = CudaValueType<T>::type;
  const cudaDataType    compute_type  = std::is_same<T, __half>::value ? CUDA_R_32F : value_type;
  const Scalar         alpha         = Scalar(1.0f);
  const Scalar         beta          = Scalar(0.0f);
  cusparseSpMatDescr_t matrix_descr  = nullptr;
  cusparseDnVecDescr_t x_descr       = nullptr;
  cusparseDnVecDescr_t y_descr       = nullptr;
  cusparseStatus_t     status        = CUSPARSE_STATUS_SUCCESS;
  cudaError_t          alloc_status  = cudaSuccess;
  size_t               buffer_size   = 0;
  void*                buffer        = nullptr;
  std::vector<T>       host_y(device.rows);
  double               widened_y     = 0.0;
  const auto           multiply_once = [&]() {
    cusparseSpMV(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix_descr, x_descr, &beta, y_descr,
                 compute_type, algorithm, buffer);
  };

  row.kernel = cusparse_algorithm_name(algorithm);
  CUSPARSE_CHECK(cusparseCreateCsr(&matrix_descr, device.rows, device.cols, device.nnz, device.row_ptr,
                                   device.col_ind_32, device.values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, value_type));
  CUSPARSE_CHECK(cusparseCreateDnVec(&x_descr, device.cols, device.x, value_type));
  CUSPARSE_CHECK(cusparseCreateDnVec(&y_descr, device.rows, device.y, value_type));
  status = cusparseSpMV_bufferSize(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix_descr, x_descr,
                                   &beta, y_descr, compute_type, algorithm, &buffer_size);
  if (status == CUSPARSE_STATUS_SUCCESS && buffer_size > 0) alloc_status = cudaMalloc(&buffer, buffer_size);
  if (status == CUSPARSE_STATUS_SUCCESS && alloc_status == cudaSuccess)
    status = cusparseSpMV_preprocess(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix_descr, x_descr,
                                     &beta, y_descr, compute_type, algorithm, buffer);
  if (status == CUSPARSE_STATUS_SUCCESS && alloc_status == cudaSuccess)
    status = cusparseSpMV(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrix_descr, x_descr, &beta,
                          y_descr, compute_type, algorithm, buffer);

  row.workspace_bytes = (double)buffer_size;
  if (status != CUSPARSE_STATUS_SUCCESS || alloc_status != cudaSuccess) {
    row.status = alloc_status != cudaSuccess ? "workspace allocation failed" : cusparseGetErrorString(status);
    std::printf("  %-4s %-21s %s\n", row.precision.c_str(), row.kernel.c_str(), row.status.c_str());
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_y.data(), device.y, sizeof(T) * device.rows, cudaMemcpyDeviceToHost));
    for (int index = 0; index < device.rows; ++index) {
      widened_y        = (double)widen_value(host_y[index]);
      row.max_abs_diff = std::max(row.max_abs_diff, std::fabs(widened_y - reference_y[index]));
    }
    row.max_rel_diff = reference_scale > 0.0 ? row.max_abs_diff / reference_scale : row.max_abs_diff;
    for (int warmup_rep = 0; warmup_rep < args.warmup; ++warmup_rep) multiply_once();
    CUDA_CHECK(cudaDeviceSynchronize());
    row.batch             = batch_for(multiply_once, args);
    row.time_ms           = batched_median_ms(multiply_once, args, row.batch);
    row.gflops            = 2.0 * device.nnz / (row.time_ms * 1e-3) / 1e9;
    row.bw_gbs            = row.memory.total / (row.time_ms * 1e-3) / 1e9;
    row.epsilon           = row.bw_gbs / read_bandwidth;
    row.speedup_measured  = baseline_ms / row.time_ms;
    row.speedup_predicted = 1.0;
    std::printf("  %-4s %-21s     %8.3f %8.2f %6.3f %8.3f %8.3f %9.1e  (workspace %.1f MiB)\n",
                row.precision.c_str(), row.kernel.c_str(), row.time_ms, row.gflops, row.epsilon,
                row.speedup_predicted, row.speedup_measured, row.max_rel_diff, buffer_size / 1048576.0);
  }
  append_row(args.csv_path, row);
  if (buffer) cudaFree(buffer);
  cusparseDestroyDnVec(x_descr);
  cusparseDestroyDnVec(y_descr);
  cusparseDestroySpMat(matrix_descr);
}

template <class T>
static void run_precision(cusparseHandle_t sparse_handle, const CsrMatrix& matrix, const MatrixFacts& facts,
                          const BenchArgs& args, double read_bandwidth) {
  constexpr int         value_bytes       = sizeof(T);
  const Configuration   baseline          = {value_bytes, 4, ColumnMode::absolute};
  const double          baseline_bytes    = memory_of(matrix, facts, baseline).total;
  const double          needed_bytes      = device_bytes_needed(matrix, facts, value_bytes);
  DeviceMatrix<T>       device;
  Configuration         active;
  std::vector<double>   reference_y;
  std::vector<double>   check_reference_y;
  std::vector<double>   result;
  int                   check_tpr         = 8;
  double                baseline_ms       = 0.0;
  double                reference_scale   = 0.0;
  ResultRow             row;
  ResultRow             baseline_row;
  size_t                free_bytes        = 0;
  size_t                total_bytes       = 0;
  double                lane_packed_ms    = 0.0;
  int                   lane_packed_tpr   = 0;
  int                   lane_packed_batch = 1;
  std::vector<ThreadSample> samples;
  std::vector<ThreadSample>* wanted_samples = args.tpr_csv_path.empty() ? nullptr : &samples;

  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (needed_bytes > 0.9 * free_bytes) {
    std::printf("  %s: skipped, needs %.2f GiB of device memory, %.2f GiB free\n", PrecisionName<T>::text,
                needed_bytes / 1073741824.0, free_bytes / 1073741824.0);
    return;
  }
  upload_matrix(matrix, facts, device);
  reference_y = result_at_fixed_tpr<T>(baseline, device, 8);
  for (double reference_value : reference_y) reference_scale = std::max(reference_scale, std::fabs(reference_value));

  for (const Configuration& configuration : configurations_for()) {
    if (configuration.value_bytes != value_bytes) continue;
    active = configuration;
    // The lane bit-packed stream depends on the threads per row, so its sweep rebuilds the format and
    // has to run before the byte count and the admissibility test, both of which read that choice.
    lane_packed_ms = 0.0;
    if (active.column_mode == ColumnMode::lane_packed) {
      lane_packed_ms   = best_lane_packed_time<T>(matrix, facts, active, device, args, lane_packed_tpr,
                                                  lane_packed_batch, wanted_samples);
      active.fixed_tpr = lane_packed_tpr > 0 ? lane_packed_tpr : 8;
    }
    row = ResultRow{};
    row.matrix       = matrix.name;
    row.ordering     = args.ordering;
    row.precision    = PrecisionName<T>::text;
    row.encoding     = "cumulative";
    row.column_mode  = column_mode_name(active.column_mode);
    row.value_bytes  = value_bytes;
    row.column_bytes = active.column_bytes;
    row.length_bytes = 4;
    row.rows         = matrix.rows;
    row.cols         = matrix.cols;
    row.nnz          = matrix.nnz();
    row.memory       = memory_of(matrix, facts, active);
    if ((active.column_mode == ColumnMode::lane_packed && lane_packed_tpr == 0) ||
        !configuration_admissible(matrix, facts, active)) {
      row.status = "not admissible";
      append_row(args.csv_path, row);
      continue;
    }
    // best_lane_packed_time already left the winning stream on the device.
    check_tpr         = active.fixed_tpr > 0 ? active.fixed_tpr : 8;
    check_reference_y = result_at_fixed_tpr<T>(baseline, device, check_tpr);
    result            = result_at_fixed_tpr<T>(active, device, check_tpr);
    for (int index = 0; index < matrix.rows; ++index)
      row.max_abs_diff = std::max(row.max_abs_diff, std::fabs(result[index] - check_reference_y[index]));
    row.max_rel_diff = reference_scale > 0.0 ? row.max_abs_diff / reference_scale : row.max_abs_diff;
    if (active.column_mode == ColumnMode::lane_packed) {
      row.time_ms  = lane_packed_ms;
      row.best_tpr = lane_packed_tpr;
      row.batch    = lane_packed_batch;
    } else {
      row.time_ms = best_time<T>(active, device, args, row.best_tpr, row.batch, wanted_samples);
      // Every candidate of these layouts runs on the same arrays, so the bytes are those of the row and
      // the check only needs the launch at that threads per row.
      for (ThreadSample& sample : samples) {
        sample.bytes_total  = row.memory.total;
        sample.max_abs_diff = wanted_samples ? max_difference_at<T>(active, device, sample.threads_per_row) : 0.0;
      }
    }
    if (wanted_samples) append_thread_samples(args.tpr_csv_path, row, column_label(active), samples);
    if (active.column_bytes == 4 && active.column_mode == ColumnMode::absolute) {
      baseline_ms  = row.time_ms;
      baseline_row = row;
    }
    row.gflops            = 2.0 * matrix.nnz() / (row.time_ms * 1e-3) / 1e9;
    row.bw_gbs            = row.memory.total / (row.time_ms * 1e-3) / 1e9;
    row.epsilon           = row.bw_gbs / read_bandwidth;
    row.speedup_measured  = baseline_ms / row.time_ms;
    row.speedup_predicted = baseline_bytes / row.memory.total;
    std::printf("  %-4s %-10s %-3s %-5s %3d %8.3f %8.2f %6.3f %8.3f %8.3f %9.1e\n", row.precision.c_str(),
                row.encoding.c_str(), column_label(active).c_str(), "ptr32", row.best_tpr,
                row.time_ms, row.gflops, row.epsilon, row.speedup_predicted, row.speedup_measured, row.max_abs_diff);
    append_row(args.csv_path, row);
    if (active.column_mode == ColumnMode::lane_packed) {
      cudaFree(device.lane_row_word);
      cudaFree(device.lane_row_width);
      cudaFree(device.lane_words);
      device.lane_row_word  = nullptr;
      device.lane_row_width = nullptr;
      device.lane_words     = nullptr;
    }
  }
  baseline_row.best_tpr = 0;
  baseline_row.time_ms  = 0;
  run_cusparse<T>(sparse_handle, CUSPARSE_SPMV_CSR_ALG1, device, args, read_bandwidth, baseline_ms, reference_y,
                  reference_scale, baseline_row);
  run_cusparse<T>(sparse_handle, CUSPARSE_SPMV_CSR_ALG2, device, args, read_bandwidth, baseline_ms, reference_y,
                  reference_scale, baseline_row);
  free_matrix(device);
}

// Gain at unchanged value precision (FP64), from the byte model.
static void print_fp64_gain(const CsrMatrix& matrix, const MatrixFacts& facts) {
  const Configuration   baseline          = {8, 4, ColumnMode::absolute};
  const MemoryBreakdown base              = memory_of(matrix, facts, baseline);
  const int             narrow_absolute   = column_width_admissible(matrix, 1) ? 1 : (column_width_admissible(matrix, 2) ? 2 : 4);
  const int             narrow_relative   = row_base_admissible(facts, 1) ? 1 : (row_base_admissible(facts, 2) ? 2 : 0);
  const Configuration   absolute_columns  = {8, narrow_absolute, ColumnMode::absolute};
  const Configuration   relative_columns  = {8, narrow_relative, ColumnMode::row_base};
  const double          nnz               = (double)matrix.nnz();
  Configuration         lane_packed       = {8, 0, ColumnMode::lane_packed};

  std::printf("  FP64 gain without reducing precision (byte model):\n");
  std::printf("    narrowest absolute columns (i%d)  %.3fx\n", narrow_absolute * 8,
              base.total / memory_of(matrix, facts, absolute_columns).total);
  if (narrow_relative)
    std::printf("    row-relative columns (r%d)        %.3fx\n", narrow_relative * 8,
                base.total / memory_of(matrix, facts, relative_columns).total);
  else
    std::printf("    row-relative columns             not admissible (widest row spans %d columns)\n",
                facts.widest_row_span);
  for (int slot = 1; slot <= 5; ++slot) {
    lane_packed.fixed_tpr = 1 << slot;
    std::printf("    lane bit-packed deltas, tpr %2d (%.2f bits/nnz) %.3fx\n", 1 << slot,
                32.0 * facts.lane_packed_words[slot] / nnz, base.total / memory_of(matrix, facts, lane_packed).total);
  }
  std::printf("    ceiling, no column indices        %.3fx\n", base.total / (base.total - base.columns));
  std::printf("    ceiling, no row pointer           %.3fx\n", base.total / (base.total - base.row_pointer));
}

static void parse_args(int argc, char** argv, BenchArgs& args) {
  std::string option;

  for (int argument = 1; argument < argc; ++argument) {
    option = argv[argument];
    if (option == "--help" || argument + 1 >= argc) {
      std::printf("usage: bench_csr_width --matrix A.mtx [--matrix B.mtx ...] [--reps 30] [--warmup 10] "
                  "[--ordering original|rcm] [--min-sample-ms 0] [--tpr-csv FILE] [--csv FILE] "
                  "[--precision all|fp64|fp32|fp16]\n");
      std::exit(option == "--help" ? 0 : 1);
    }
    if      (option == "--matrix")   args.matrix_paths.push_back(argv[++argument]);
    else if (option == "--csv")      args.csv_path = argv[++argument];
    else if (option == "--reps")     args.reps     = std::stoi(argv[++argument]);
    else if (option == "--warmup")   args.warmup   = std::stoi(argv[++argument]);
    else if (option == "--ordering") args.ordering = argv[++argument];
    else if (option == "--min-sample-ms") args.min_sample_ms = std::stod(argv[++argument]);
    else if (option == "--tpr-csv")  args.tpr_csv_path = argv[++argument];
    else if (option == "--precision") args.precision = argv[++argument];
    else { std::fprintf(stderr, "unknown option %s\n", option.c_str()); std::exit(1); }
  }
  if (args.matrix_paths.empty()) { std::fprintf(stderr, "at least one --matrix is required\n"); std::exit(1); }
  if (args.ordering != "original" && args.ordering != "rcm") {
    std::fprintf(stderr, "--ordering must be original or rcm\n");
    std::exit(1);
  }
  if (args.precision != "all" && args.precision != "fp64" && args.precision != "fp32" && args.precision != "fp16") {
    std::fprintf(stderr, "--precision must be all, fp64, fp32 or fp16\n");
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  BenchArgs   args;
  DeviceInfo  device;
  double      read_bandwidth = 0.0;
  CsrMatrix        matrix;
  MatrixFacts      facts;
  cusparseHandle_t sparse_handle  = nullptr;
  double           ordering_seconds = 0.0;

  parse_args(argc, argv, args);
  device         = query_device(0);
  CUSPARSE_CHECK(cusparseCreate(&sparse_handle));
  read_bandwidth = measure_read_bandwidth(device);
  std::printf("device %s (cc %d.%d), read bandwidth %.1f GB/s, L2 %.1f MiB\n", device.name.c_str(),
              device.cc_major, device.cc_minor, read_bandwidth, device.l2_bytes / 1048576.0);

  for (const std::string& matrix_path : args.matrix_paths) {
    try {
      matrix = read_matrix_market(matrix_path);
      if (args.ordering == "rcm") matrix = reorder_with_rcm(matrix_path, matrix, ordering_seconds);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "[skip] %s\n", error.what());
      continue;
    }
    facts = facts_of(matrix);
    std::printf("\n=== %s: m=%d n=%d nnz=%lld, %.1f nnz/row (max %d) ===\n", matrix.name.c_str(), matrix.rows,
                matrix.cols, (long long)matrix.nnz(), matrix.stats.nnz_per_row_mean, matrix.stats.nnz_per_row_max);
    if (args.ordering == "rcm")
      std::printf("  ordering: rcm, symmetric permutation P A P^T (ordering computed in %.2f s)\n", ordering_seconds);
    std::printf("  minimum bits: column index %d, cumulative row pointer %d, row length %d, widest row span %d (%d bits)\n",
                required_bits((uint64_t)matrix.cols - 1), required_bits((uint64_t)matrix.nnz()),
                required_bits((uint64_t)matrix.stats.nnz_per_row_max), facts.widest_row_span,
                required_bits((uint64_t)facts.widest_row_span));
    print_memory_table(matrix, facts);
    print_fp64_gain(matrix, facts);
    std::printf("  %-4s %-10s %-3s %-5s %3s %8s %8s %6s %8s %8s %9s\n", "prec", "row ptr", "col", "len",
                "tpr", "t[ms]", "GFLOP/s", "eps", "pred", "meas", "max|dy|");
#ifndef SPMV_HALF_ACCUMULATE
    if (args.precision == "all" || args.precision == "fp64")
      run_precision<double>(sparse_handle, matrix, facts, args, read_bandwidth);
    if (args.precision == "all" || args.precision == "fp32")
      run_precision<float>(sparse_handle, matrix, facts, args, read_bandwidth);
#endif
    if (args.precision == "all" || args.precision == "fp16")
      run_precision<__half>(sparse_handle, matrix, facts, args, read_bandwidth);
  }
  cusparseDestroy(sparse_handle);
  std::printf("\nresults appended to %s\n", args.csv_path.c_str());
  return 0;
}
