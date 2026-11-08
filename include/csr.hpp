#pragma once
// csr.hpp — host-side CSR container + Matrix Market reader (symmetric expansion)
//           + the structural statistics that predict format behaviour.
//
// The stats block is not decoration: nnz/row std and max are exactly what
// decides whether ELL is viable, and dynamic_range is what predicts fp16
// failure. Both are logged for every matrix.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct MatrixStats {
  double nnz_per_row_mean = 0.0;
  int    nnz_per_row_max  = 0;
  int    nnz_per_row_min  = 0;
  double nnz_per_row_std  = 0.0;
  double dynamic_range    = 0.0;   // max|a_ij| / min|a_ij|  (nonzeros only)
  double norm_inf         = 0.0;   // max_i sum_j |a_ij|
  double max_abs          = 0.0;
  double min_abs          = 0.0;
};

struct CsrMatrix {
  int                  rows = 0;
  int                  cols = 0;
  std::vector<int>     row_ptr;   // size rows+1
  std::vector<int>     col_ind;   // size nnz
  std::vector<double>  val;       // size nnz
  MatrixStats          stats;
  bool                 symmetric_expanded = false;
  std::string          name;

  int64_t nnz() const { return (int64_t)col_ind.size(); }

  void compute_stats() {
    double row_length_sum         = 0.0;
    double row_length_square_sum  = 0.0;
    double largest_abs_value      = 0.0;
    double smallest_abs_value     = 1e308;
    int    row_length             = 0;
    double row_abs_sum            = 0.0;
    double abs_value              = 0.0;

    stats = MatrixStats{};
    if (rows == 0) return;
    stats.nnz_per_row_min = INT32_MAX;
    for (int row = 0; row < rows; ++row) {
      row_length = row_ptr[row + 1] - row_ptr[row];
      row_length_sum        += row_length;
      row_length_square_sum += (double)row_length * row_length;
      stats.nnz_per_row_max = std::max(stats.nnz_per_row_max, row_length);
      stats.nnz_per_row_min = std::min(stats.nnz_per_row_min, row_length);
      row_abs_sum = 0.0;
      for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry) {
        abs_value = std::fabs(val[entry]);
        row_abs_sum += abs_value;
        if (abs_value > 0.0) {
          largest_abs_value  = std::max(largest_abs_value, abs_value);
          smallest_abs_value = std::min(smallest_abs_value, abs_value);
        }
      }
      stats.norm_inf = std::max(stats.norm_inf, row_abs_sum);
    }
    stats.nnz_per_row_mean = row_length_sum / rows;
    stats.nnz_per_row_std  = std::sqrt(std::max(
        0.0, row_length_square_sum / rows - stats.nnz_per_row_mean * stats.nnz_per_row_mean));
    stats.max_abs = largest_abs_value;
    stats.min_abs = (smallest_abs_value < 1e308) ? smallest_abs_value : 0.0;
    stats.dynamic_range = (stats.min_abs > 0.0) ? stats.max_abs / stats.min_abs : 0.0;
  }

  void print_stats(std::FILE* out = stdout) const {
    std::fprintf(out, "matrix      : %s\n", name.c_str());
    std::fprintf(out, "dims        : %d x %d, nnz = %lld%s\n", rows, cols,
                 (long long)nnz(), symmetric_expanded ? " (symmetric, expanded)" : "");
    std::fprintf(out, "nnz/row     : mean %.2f  min %d  max %d  std %.2f\n",
                 stats.nnz_per_row_mean, stats.nnz_per_row_min,
                 stats.nnz_per_row_max, stats.nnz_per_row_std);
    std::fprintf(out, "dyn range   : %.3e   (max|a| %.3e / min|a| %.3e)\n",
                 stats.dynamic_range, stats.max_abs, stats.min_abs);
    std::fprintf(out, "||A||_inf   : %.3e\n", stats.norm_inf);
  }
};

// ---------------------------------------------------------------------------
// Matrix Market reader
// ---------------------------------------------------------------------------
// Handles: coordinate real/integer/pattern, general/symmetric/skew-symmetric.
// Symmetric files are expanded to full storage -- SpMV kernels here do not
// exploit symmetry, and the format comparison must see the true row lengths.

struct MatrixMarketEntry { int row, col; double value; };

// Tokens of the banner line "%%MatrixMarket matrix coordinate <field> <symmetry>".
struct MatrixMarketBanner {
  std::string tag, object, format, field, symmetry;
};

inline MatrixMarketBanner parse_banner(const std::string& banner_line) {
  MatrixMarketBanner banner;
  std::istringstream banner_stream(banner_line);

  banner_stream >> banner.tag >> banner.object >> banner.format >> banner.field >> banner.symmetry;
  for (std::string* token : {&banner.object, &banner.format, &banner.field, &banner.symmetry})
    std::transform(token->begin(), token->end(), token->begin(), ::tolower);
  return banner;
}

// Reads "rows cols nnz" from the size line.
inline void parse_size_line(const std::string& size_line, int& rows, int& cols, int64_t& nnz) {
  std::istringstream size_stream(size_line);
  size_stream >> rows >> cols >> nnz;
}

// Advances the stream past the banner and every comment line, leaving it just
// after the size line, whose text is returned in size_line.
inline void skip_to_body(std::ifstream& file, std::string& size_line) {
  file.clear();
  file.seekg(0, std::ios::beg);
  std::getline(file, size_line);                       // banner
  while (std::getline(file, size_line))
    if (!size_line.empty() && size_line[0] != '%') break;
}

// Parses the coordinate body in one pass over a single in-memory buffer:
// getline per entry is the bottleneck on the 70 MB files in the collection.
// The buffer is local to this function so it is released as soon as parsing ends.
inline void parse_body(std::ifstream& file, int rows, int cols, int64_t declared_nnz,
                       bool is_pattern, bool is_symmetric, bool is_skew,
                       std::vector<MatrixMarketEntry>& entries) {
  std::ostringstream body_buffer;
  std::string        body_text;
  const char*        cursor      = nullptr;
  const char*        body_end    = nullptr;
  char*              parse_stop  = nullptr;
  long               row_one_based = 0;
  long               col_one_based = 0;
  double             value       = 1.0;
  int                row_index   = 0;
  int                col_index   = 0;

  body_buffer << file.rdbuf();
  body_text = body_buffer.str();
  cursor    = body_text.c_str();
  body_end  = cursor + body_text.size();

  for (int64_t entry = 0; entry < declared_nnz && cursor < body_end; ++entry) {
    row_one_based = std::strtol(cursor, &parse_stop, 10);
    if (parse_stop == cursor) break;
    cursor = parse_stop;
    col_one_based = std::strtol(cursor, &parse_stop, 10);
    cursor = parse_stop;
    value = 1.0;
    if (!is_pattern) {
      value  = std::strtod(cursor, &parse_stop);
      cursor = parse_stop;
    }
    row_index = (int)row_one_based - 1;
    col_index = (int)col_one_based - 1;
    while (cursor < body_end &&
           (*cursor == '\n' || *cursor == '\r' || *cursor == ' ' || *cursor == '\t'))
      ++cursor;
    if (row_index < 0 || row_index >= rows || col_index < 0 || col_index >= cols) continue;
    entries.push_back({row_index, col_index, value});
    if (is_symmetric && row_index != col_index)
      entries.push_back({col_index, row_index, is_skew ? -value : value});
  }
}

// Scatters the triplets into CSR arrays already sized by the caller.
inline void scatter_entries(const std::vector<MatrixMarketEntry>& entries, CsrMatrix& matrix) {
  std::vector<int> next_slot_in_row(matrix.row_ptr.begin(), matrix.row_ptr.end() - 1);
  int              destination = 0;

  for (const MatrixMarketEntry& entry : entries) {
    destination = next_slot_in_row[entry.row]++;
    matrix.col_ind[destination] = entry.col;
    matrix.val[destination]     = entry.value;
  }
}

// Sort columns within each row. Required for a fair comparison: unsorted
// column indices penalise every format's cache behaviour identically in
// principle, but ELL/HLL read x in a different order than CSR does, so an
// unsorted input would silently bias the format ranking.
inline void sort_columns_within_rows(CsrMatrix& matrix) {
  std::vector<int>    order;
  std::vector<int>    sorted_columns;
  std::vector<double> sorted_values;
  int                 row_begin = 0;
  int                 row_end   = 0;

  for (int row = 0; row < matrix.rows; ++row) {
    row_begin = matrix.row_ptr[row];
    row_end   = matrix.row_ptr[row + 1];
    order.resize(row_end - row_begin);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int left, int right) {
      return matrix.col_ind[row_begin + left] < matrix.col_ind[row_begin + right];
    });
    sorted_columns.resize(order.size());
    sorted_values.resize(order.size());
    for (size_t position = 0; position < order.size(); ++position) {
      sorted_columns[position] = matrix.col_ind[row_begin + order[position]];
      sorted_values[position]  = matrix.val[row_begin + order[position]];
    }
    std::copy(sorted_columns.begin(), sorted_columns.end(), matrix.col_ind.begin() + row_begin);
    std::copy(sorted_values.begin(),  sorted_values.end(),  matrix.val.begin() + row_begin);
  }
}

// File name without directory and extension, used as the matrix label.
inline std::string matrix_name_from_path(const std::string& path) {
  size_t      last_separator = path.find_last_of("/\\");
  std::string file_name      = (last_separator == std::string::npos) ? path
                                                                     : path.substr(last_separator + 1);
  size_t      last_dot       = file_name.find_last_of('.');

  return (last_dot == std::string::npos) ? file_name : file_name.substr(0, last_dot);
}

inline CsrMatrix read_matrix_market(const std::string& path) {
  std::ifstream                  file(path);
  std::string                    line;
  MatrixMarketBanner             banner;
  bool                           is_pattern   = false;
  bool                           is_symmetric = false;
  bool                           is_skew      = false;
  int                            rows         = 0;
  int                            cols         = 0;
  int64_t                        declared_nnz = 0;
  std::vector<MatrixMarketEntry> entries;
  CsrMatrix                      matrix;

  if (!file) throw std::runtime_error("cannot open matrix file: " + path);
  if (!std::getline(file, line)) throw std::runtime_error("empty matrix file: " + path);
  if (line.rfind("%%MatrixMarket", 0) != 0)
    throw std::runtime_error("not a Matrix Market file: " + path);

  banner = parse_banner(line);
  if (banner.format != "coordinate")
    throw std::runtime_error("only coordinate format supported (got " + banner.format + ")");
  is_pattern   = (banner.field == "pattern");
  is_symmetric = (banner.symmetry == "symmetric" || banner.symmetry == "skew-symmetric" ||
                  banner.symmetry == "hermitian");
  is_skew      = (banner.symmetry == "skew-symmetric");

  skip_to_body(file, line);
  parse_size_line(line, rows, cols, declared_nnz);
  if (rows <= 0 || cols <= 0) throw std::runtime_error("bad dimensions in " + path);

  entries.reserve(is_symmetric ? (size_t)declared_nnz * 2 : (size_t)declared_nnz);
  parse_body(file, rows, cols, declared_nnz, is_pattern, is_symmetric, is_skew, entries);

  matrix.rows = rows;
  matrix.cols = cols;
  matrix.symmetric_expanded = is_symmetric;
  matrix.row_ptr.assign(rows + 1, 0);
  for (const MatrixMarketEntry& entry : entries) matrix.row_ptr[entry.row + 1]++;
  for (int row = 0; row < rows; ++row) matrix.row_ptr[row + 1] += matrix.row_ptr[row];
  matrix.col_ind.resize(entries.size());
  matrix.val.resize(entries.size());

  scatter_entries(entries, matrix);
  sort_columns_within_rows(matrix);
  matrix.name = matrix_name_from_path(path);
  matrix.compute_stats();
  return matrix;
}
