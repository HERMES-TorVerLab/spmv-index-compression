// characterize_matrix.cpp — structural metrics that drive the index-compression
// byte model, and the speedup ceiling each matrix admits.
//
// Host-only. One matrix per invocation, one CSV line (with header) per output
// file, so many invocations can run in parallel without sharing a file.
//
// All byte quantities are for FP64 values (v = 8) and 32-bit row pointers
// (p = 4), the scientific-computing configuration this study fixes. With
// m rows, n columns and a column-index width of i bytes, the idealised traffic
// per nonzero is
//
//   q = v + i + [ (m+1)p + (n+m)v ] / nnz  +  (base overhead) / nnz
//
// and a layout's speedup over CSR-i32 predicted by the byte model is
// q_csr_i32 / q_layout. The ceiling for ANY index compression is obtained by
// removing the column-index stream altogether (i = 0, no bases):
//
//   S_max = q_csr_i32 / ( v + [ (m+1)p + (n+m)v ] / nnz )

#include "csr.hpp"
#include "rcm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

struct Characterization {
  std::string name;
  int         rows                 = 0;
  int         cols                 = 0;
  int64_t     nnz                  = 0;
  double      row_length_mean      = 0.0;
  int         row_length_max       = 0;
  double      row_length_std       = 0.0;
  int64_t     empty_rows           = 0;
  int64_t     bandwidth            = 0;     // max |row - col|
  int         row_span_max         = 0;     // max over rows of (last col - first col)
  double      row_span_mean        = 0.0;
  int         row_span_p50         = 0;
  int         row_span_p90         = 0;
  double      rows_fitting_8bit    = 0.0;   // fraction of rows with span < 256
  double      rows_fitting_16bit   = 0.0;   // fraction of rows with span < 65536
  double      nnz_in_rows_fitting_8bit  = 0.0;
  double      nnz_in_rows_fitting_16bit = 0.0;
  int64_t     segments_8bit        = 0;     // windowed layout segment count
  int64_t     segments_16bit       = 0;
  double      gamma_csr_i32        = 0.0;
  double      bytes_per_nnz_csr_i32      = 0.0;
  double      bytes_per_nnz_index_free   = 0.0;
  double      bytes_per_nnz_rowbase_16   = NAN;   // NAN when some row does not fit
  double      bytes_per_nnz_rowbase_8    = NAN;
  double      bytes_per_nnz_windowed_16  = 0.0;
  double      bytes_per_nnz_windowed_8   = 0.0;
  double      speedup_ceiling            = 0.0;
  double      speedup_rowbase_16         = NAN;
  double      speedup_rowbase_8          = NAN;
  double      speedup_windowed_16        = 0.0;
  double      speedup_windowed_8         = 0.0;
  double      working_set_mib_csr_i32    = 0.0;
  double      dynamic_range              = 0.0;
  std::string ordering                   = "original";   // original | rcm (symmetric P A P^T)
  double      ordering_seconds           = 0.0;          // time to compute the ordering
  // Work-side regularity of the row-length distribution, for a vector kernel with T threads per row.
  double      lane_occupancy[5]          = {0.0, 0.0, 0.0, 0.0, 0.0};   // T = 2, 4, 8, 16, 32
  double      tail_ratio[5]              = {0.0, 0.0, 0.0, 0.0, 0.0};
  // Traffic of the gather of x, counted in sectors rather than in elements, for v = 8, 4 and 2 bytes
  // and for a cache of each capacity in CACHE_SECTORS. The first entry keeps nothing, so it is the
  // bound with no reuse at all.
  double      gather_bytes[3][5]         = {{0.0}};
};

static const int VALUE_BYTES[3]   = {8, 4, 2};
static const int SECTOR_BYTES     = 32;
// Capacities in sectors: none, 8, 32, 128 and the whole 4 MiB L2. The launch keeps one vector per row
// and the device holds N/T of them at once, so the share of the L2 a row in flight can call its own is
// between 5 sectors at T = 2 and 85 at T = 32, which the three small capacities bracket. Which one
// reproduces the measured traffic is then a test of that derivation rather than a fitted coefficient.
static const int64_t CACHE_SECTORS[5] = {0, 8, 32, 128, 131072};

static const int THREADS_PER_ROW[5] = {2, 4, 8, 16, 32};

// Traffic of the gather of x, counted the way the memory system moves it. The byte model charges one
// element of v bytes per nonzero, but a memory access pulls a whole sector, so a scattered row pays for
// sectors and not for elements. Columns inside a row are sorted, so the sectors a row touches are the
// distinct values of (column * v) / SECTOR_BYTES along the row, which one scan finds without a set.
// The same scan feeds caches of several capacities, each a first-in first-out set of sectors, so that
// the analysis can ask which capacity the device behaves as if it had rather than fit a coefficient.
static void measure_gather_traffic(const CsrMatrix& matrix, Characterization& result) {
  std::unordered_set<int64_t> resident[5];
  std::vector<int64_t>        eviction_order[5];
  int64_t                     misses[5]   = {0, 0, 0, 0, 0};
  int64_t                     oldest[5]   = {0, 0, 0, 0, 0};
  int64_t                     sector      = 0;
  int64_t                     previous    = -1;
  int                         value_bytes = 0;

  for (int index = 0; index < 3; ++index) {
    value_bytes = VALUE_BYTES[index];
    for (int level = 0; level < 5; ++level) {
      misses[level] = 0;
      oldest[level] = 0;
      resident[level].clear();
      eviction_order[level].clear();
    }
    for (int row = 0; row < matrix.rows; ++row) {
      previous = -1;
      for (int entry = matrix.row_ptr[row]; entry < matrix.row_ptr[row + 1]; ++entry) {
        sector = (int64_t)matrix.col_ind[entry] * value_bytes / SECTOR_BYTES;
        if (sector == previous) continue;
        previous = sector;
        ++misses[0];
        for (int level = 1; level < 5; ++level) {
          if (resident[level].find(sector) != resident[level].end()) continue;
          ++misses[level];
          resident[level].insert(sector);
          eviction_order[level].push_back(sector);
          if ((int64_t)eviction_order[level].size() - oldest[level] > CACHE_SECTORS[level]) {
            resident[level].erase(eviction_order[level][oldest[level]]);
            ++oldest[level];
          }
        }
      }
    }
    for (int level = 0; level < 5; ++level)
      result.gather_bytes[index][level] = (double)misses[level] * SECTOR_BYTES / (double)matrix.nnz();
  }
}

// Regularity of the row-length distribution, as a vector kernel with T threads per row sees it.
// A row of length l needs ceil(l / T) iterations and therefore occupies T * ceil(l / T) lane slots to
// perform l multiplications, so the occupancy is the share of issued slots that do useful work and it
// falls when rows are shorter than T. The tail ratio is the longest row in units of the mean row, which
// is what a single huge row costs once every other row has finished. Neither quantity moves a byte, so
// they belong to the work term and not to the byte count.
static void measure_lane_occupancy(const CsrMatrix& matrix, Characterization& result) {
  int64_t occupied_slots  = 0;
  int64_t total_iterations = 0;
  int64_t longest_row      = 0;
  int64_t row_iterations   = 0;
  int     threads          = 0;
  int     row_length       = 0;

  for (int index = 0; index < 5; ++index) {
    threads          = THREADS_PER_ROW[index];
    occupied_slots   = 0;
    total_iterations = 0;
    longest_row      = 0;
    for (int row = 0; row < matrix.rows; ++row) {
      row_length       = matrix.row_ptr[row + 1] - matrix.row_ptr[row];
      row_iterations   = (row_length + threads - 1) / threads;
      occupied_slots  += row_iterations * threads;
      total_iterations += row_iterations;
      longest_row      = std::max(longest_row, row_iterations);
    }
    result.lane_occupancy[index] = occupied_slots > 0 ? (double)matrix.nnz() / (double)occupied_slots : 0.0;
    result.tail_ratio[index]     = total_iterations > 0
                                 ? (double)longest_row * (double)matrix.rows / (double)total_iterations
                                 : 0.0;
  }
}

// Segments needed when each row is cut into pieces of column span < window.
static int64_t count_windowed_segments(const CsrMatrix& matrix, int64_t window) {
  int64_t segments     = 0;
  int     current_base = 0;
  int     column       = 0;

  for (int row = 0; row < matrix.rows; ++row) {
    for (int entry = matrix.row_ptr[row]; entry < matrix.row_ptr[row + 1]; ++entry) {
      column = matrix.col_ind[entry];
      if (entry == matrix.row_ptr[row] || (int64_t)column - current_base >= window) {
        current_base = column;
        ++segments;
      }
    }
  }
  return segments;
}

// Row spans, bandwidth and the fraction of rows / nonzeros that fit each window.
static void measure_row_spans(const CsrMatrix& matrix, Characterization& result) {
  std::vector<int> spans(matrix.rows, 0);
  int64_t          fitting_rows_8   = 0;
  int64_t          fitting_rows_16  = 0;
  int64_t          fitting_nnz_8    = 0;
  int64_t          fitting_nnz_16   = 0;
  double           span_sum         = 0.0;
  int              row_begin        = 0;
  int              row_end          = 0;
  int              row_length       = 0;
  int64_t          distance         = 0;

  for (int row = 0; row < matrix.rows; ++row) {
    row_begin  = matrix.row_ptr[row];
    row_end    = matrix.row_ptr[row + 1];
    row_length = row_end - row_begin;
    if (row_length == 0) {
      ++result.empty_rows;
    } else {
      spans[row] = matrix.col_ind[row_end - 1] - matrix.col_ind[row_begin];
      distance   = std::max<int64_t>(std::llabs((int64_t)row - matrix.col_ind[row_begin]),
                                     std::llabs((int64_t)row - matrix.col_ind[row_end - 1]));
      result.bandwidth = std::max(result.bandwidth, distance);
    }
    span_sum += spans[row];
    if (spans[row] < 256)   { ++fitting_rows_8;  fitting_nnz_8  += row_length; }
    if (spans[row] < 65536) { ++fitting_rows_16; fitting_nnz_16 += row_length; }
  }
  result.rows_fitting_8bit         = (double)fitting_rows_8 / matrix.rows;
  result.rows_fitting_16bit        = (double)fitting_rows_16 / matrix.rows;
  result.nnz_in_rows_fitting_8bit  = (double)fitting_nnz_8 / (double)matrix.nnz();
  result.nnz_in_rows_fitting_16bit = (double)fitting_nnz_16 / (double)matrix.nnz();
  result.row_span_mean             = span_sum / matrix.rows;
  std::sort(spans.begin(), spans.end());
  result.row_span_max = spans.back();
  result.row_span_p50 = spans[(size_t)(0.50 * (spans.size() - 1))];
  result.row_span_p90 = spans[(size_t)(0.90 * (spans.size() - 1))];
}

// FP64 byte model for every layout, and the predicted speedups.
static void apply_byte_model(const CsrMatrix& matrix, Characterization& result) {
  const double value_bytes      = 8.0;
  const double row_ptr_bytes    = 4.0;
  const double nnz              = (double)matrix.nnz();
  const double auxiliary_bytes  = ((matrix.rows + 1.0) * row_ptr_bytes
                                   + (double)(matrix.cols + matrix.rows) * value_bytes) / nnz;
  const double row_base_bytes   = 4.0 * matrix.rows / nnz;
  const double segment_bytes_8  = ((result.segments_8bit + 1.0) * 4.0 + 4.0 * result.segments_8bit) / nnz;
  const double segment_bytes_16 = ((result.segments_16bit + 1.0) * 4.0 + 4.0 * result.segments_16bit) / nnz;

  result.bytes_per_nnz_csr_i32     = value_bytes + 4.0 + auxiliary_bytes;
  result.bytes_per_nnz_index_free  = value_bytes + auxiliary_bytes;
  result.bytes_per_nnz_windowed_16 = value_bytes + 2.0 + auxiliary_bytes + segment_bytes_16;
  result.bytes_per_nnz_windowed_8  = value_bytes + 1.0 + auxiliary_bytes + segment_bytes_8;
  if (result.rows_fitting_16bit == 1.0)
    result.bytes_per_nnz_rowbase_16 = value_bytes + 2.0 + auxiliary_bytes + row_base_bytes;
  if (result.rows_fitting_8bit == 1.0)
    result.bytes_per_nnz_rowbase_8  = value_bytes + 1.0 + auxiliary_bytes + row_base_bytes;

  result.gamma_csr_i32        = result.bytes_per_nnz_csr_i32 / (value_bytes + 4.0);
  result.speedup_ceiling      = result.bytes_per_nnz_csr_i32 / result.bytes_per_nnz_index_free;
  result.speedup_windowed_16  = result.bytes_per_nnz_csr_i32 / result.bytes_per_nnz_windowed_16;
  result.speedup_windowed_8   = result.bytes_per_nnz_csr_i32 / result.bytes_per_nnz_windowed_8;
  result.speedup_rowbase_16   = result.bytes_per_nnz_csr_i32 / result.bytes_per_nnz_rowbase_16;
  result.speedup_rowbase_8    = result.bytes_per_nnz_csr_i32 / result.bytes_per_nnz_rowbase_8;
  result.working_set_mib_csr_i32 = result.bytes_per_nnz_csr_i32 * nnz / 1048576.0;
}

static void write_csv(const Characterization& result, std::FILE* output) {
  std::fprintf(output,
    "name,rows,cols,nnz,row_length_mean,row_length_max,row_length_std,empty_rows,bandwidth,"
    "row_span_max,row_span_mean,row_span_p50,row_span_p90,rows_fitting_8bit,rows_fitting_16bit,"
    "nnz_in_rows_fitting_8bit,nnz_in_rows_fitting_16bit,segments_8bit,segments_16bit,"
    "gamma_csr_i32,bytes_per_nnz_csr_i32,bytes_per_nnz_index_free,bytes_per_nnz_rowbase_16,"
    "bytes_per_nnz_rowbase_8,bytes_per_nnz_windowed_16,bytes_per_nnz_windowed_8,speedup_ceiling,"
    "speedup_rowbase_16,speedup_rowbase_8,speedup_windowed_16,speedup_windowed_8,"
    "working_set_mib_csr_i32,dynamic_range,ordering,ordering_seconds,"
    "lane_occupancy_t2,lane_occupancy_t4,lane_occupancy_t8,lane_occupancy_t16,lane_occupancy_t32,"
    "tail_ratio_t2,tail_ratio_t4,tail_ratio_t8,tail_ratio_t16,tail_ratio_t32,"
    "gather_v8_c0,gather_v8_c8,gather_v8_c32,gather_v8_c128,gather_v8_cl2,"
    "gather_v4_c0,gather_v4_c8,gather_v4_c32,gather_v4_c128,gather_v4_cl2,"
    "gather_v2_c0,gather_v2_c8,gather_v2_c32,gather_v2_c128,gather_v2_cl2\n");
  std::fprintf(output,
    "%s,%d,%d,%lld,%.4f,%d,%.4f,%lld,%lld,%d,%.2f,%d,%d,%.6f,%.6f,%.6f,%.6f,%lld,%lld,"
    "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.6e,%s,%.3f,"
    "%.6f,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,"
    "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
    result.name.c_str(), result.rows, result.cols, (long long)result.nnz, result.row_length_mean,
    result.row_length_max, result.row_length_std, (long long)result.empty_rows, (long long)result.bandwidth,
    result.row_span_max, result.row_span_mean, result.row_span_p50, result.row_span_p90,
    result.rows_fitting_8bit, result.rows_fitting_16bit, result.nnz_in_rows_fitting_8bit,
    result.nnz_in_rows_fitting_16bit, (long long)result.segments_8bit, (long long)result.segments_16bit,
    result.gamma_csr_i32, result.bytes_per_nnz_csr_i32, result.bytes_per_nnz_index_free,
    result.bytes_per_nnz_rowbase_16, result.bytes_per_nnz_rowbase_8, result.bytes_per_nnz_windowed_16,
    result.bytes_per_nnz_windowed_8, result.speedup_ceiling, result.speedup_rowbase_16,
    result.speedup_rowbase_8, result.speedup_windowed_16, result.speedup_windowed_8,
    result.working_set_mib_csr_i32, result.dynamic_range, result.ordering.c_str(), result.ordering_seconds,
    result.lane_occupancy[0], result.lane_occupancy[1], result.lane_occupancy[2], result.lane_occupancy[3],
    result.lane_occupancy[4], result.tail_ratio[0], result.tail_ratio[1], result.tail_ratio[2],
    result.tail_ratio[3], result.tail_ratio[4],
    result.gather_bytes[0][0], result.gather_bytes[0][1], result.gather_bytes[0][2],
    result.gather_bytes[0][3], result.gather_bytes[0][4],
    result.gather_bytes[1][0], result.gather_bytes[1][1], result.gather_bytes[1][2],
    result.gather_bytes[1][3], result.gather_bytes[1][4],
    result.gather_bytes[2][0], result.gather_bytes[2][1], result.gather_bytes[2][2],
    result.gather_bytes[2][3], result.gather_bytes[2][4]);
}

int main(int argc, char** argv) {
  CsrMatrix        matrix;
  Characterization result;
  std::FILE*       output        = stdout;
  std::string      matrix_path;
  std::string      output_path;
  std::string      argument_text;
  bool             usage_error   = false;

  for (int argument = 1; argument < argc; ++argument) {
    argument_text = argv[argument];
    if (argument_text == "--ordering" && argument + 1 < argc) result.ordering = argv[++argument];
    else if (matrix_path.empty()) matrix_path = argument_text;
    else if (output_path.empty()) output_path = argument_text;
    else usage_error = true;
  }
  if (usage_error || matrix_path.empty() || (result.ordering != "original" && result.ordering != "rcm")) {
    std::fprintf(stderr, "usage: characterize_matrix A.mtx [out.csv] [--ordering original|rcm]\n"
                         "  rcm: characterize P A P^T; the permutation is cached as <A>.rcm.perm\n");
    return 1;
  }
  try {
    matrix = read_matrix_market(matrix_path);
    if (matrix.rows != matrix.cols || matrix.nnz() == 0) {
      std::fprintf(stderr, "[skip] %s: not square or empty\n", matrix_path.c_str());
      return 1;
    }
    if (result.ordering == "rcm") matrix = reorder_with_rcm(matrix_path, matrix, result.ordering_seconds);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[fail] %s: %s\n", matrix_path.c_str(), error.what());
    return 1;
  }

  result.name            = matrix.name;
  result.rows            = matrix.rows;
  result.cols            = matrix.cols;
  result.nnz             = matrix.nnz();
  result.row_length_mean = matrix.stats.nnz_per_row_mean;
  result.row_length_max  = matrix.stats.nnz_per_row_max;
  result.row_length_std  = matrix.stats.nnz_per_row_std;
  result.dynamic_range   = matrix.stats.dynamic_range;
  measure_row_spans(matrix, result);
  measure_lane_occupancy(matrix, result);
  measure_gather_traffic(matrix, result);
  result.segments_8bit   = count_windowed_segments(matrix, 256);
  result.segments_16bit  = count_windowed_segments(matrix, 65536);
  apply_byte_model(matrix, result);

  if (!output_path.empty()) output = std::fopen(output_path.c_str(), "w");
  if (!output) {
    std::fprintf(stderr, "[fail] cannot write %s\n", output_path.c_str());
    return 1;
  }
  write_csv(result, output);
  if (output != stdout) std::fclose(output);
  return 0;
}
