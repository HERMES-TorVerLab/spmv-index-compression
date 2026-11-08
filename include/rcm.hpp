#pragma once
// rcm.hpp — Reverse Cuthill-McKee ordering and symmetric permutation of a CSR matrix.
//
// A narrow column index stored as an offset from a per-row base is exact only if
// every row's column span fits the index window. RCM reduces the bandwidth b of the
// symmetrised pattern, and a row span is at most 2b, so reordering widens the set of
// matrices that admit a narrow index. It also changes the access pattern on x, so a
// reordered layout must be compared against CSR-i32 on the same reordered matrix;
// the original ordering is measured separately to isolate the locality effect.
//
// The permutation is applied symmetrically, B = P A P^T, so x and y stay in one index
// space and the product of the reordered matrix is a permutation of the original one.
//
// Convention: new_to_old[new_index] = old_index.
//
// Permutation file (little endian): 8-byte magic "RCMPERM1", int64 rows,
// double seconds spent computing the ordering, then rows x int32 new_to_old.

#include "csr.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr char PERMUTATION_MAGIC[8] = {'R', 'C', 'M', 'P', 'E', 'R', 'M', '1'};
static constexpr int  PSEUDO_PERIPHERAL_ITERATIONS = 8;

// Pattern of A + A^T without the diagonal, neighbours sorted and unique.
struct SymmetricPattern {
  std::vector<int64_t> adjacency_ptr;   // rows + 1
  std::vector<int>     adjacency;
};

// Columns of A^T in CSR form: for every column c, the rows r with a_rc != 0, sorted.
inline void transpose_pattern(const CsrMatrix& matrix, std::vector<int64_t>& transpose_ptr,
                              std::vector<int>& transpose_rows) {
  std::vector<int64_t> next_slot;
  int64_t              destination = 0;

  transpose_ptr.assign((size_t)matrix.cols + 1, 0);
  for (int64_t entry = 0; entry < matrix.nnz(); ++entry) ++transpose_ptr[matrix.col_ind[entry] + 1];
  for (int column = 0; column < matrix.cols; ++column) transpose_ptr[column + 1] += transpose_ptr[column];
  next_slot.assign(transpose_ptr.begin(), transpose_ptr.end() - 1);
  transpose_rows.resize((size_t)matrix.nnz());
  for (int row = 0; row < matrix.rows; ++row) {
    for (int entry = matrix.row_ptr[row]; entry < matrix.row_ptr[row + 1]; ++entry) {
      destination = next_slot[matrix.col_ind[entry]]++;
      transpose_rows[destination] = row;
    }
  }
}

// Merges row r of A with row r of A^T. With count_only the neighbours are counted,
// otherwise they are written from output onwards. Returns the neighbour count.
inline int64_t merge_row_neighbours(const CsrMatrix& matrix, const std::vector<int64_t>& transpose_ptr,
                                    const std::vector<int>& transpose_rows, int row, bool count_only,
                                    int* output) {
  int64_t row_cursor       = matrix.row_ptr[row];
  int64_t row_end          = matrix.row_ptr[row + 1];
  int64_t transpose_cursor = transpose_ptr[row];
  int64_t transpose_end    = transpose_ptr[row + 1];
  int64_t written          = 0;
  int     neighbour        = 0;
  int     previous         = -1;

  while (row_cursor < row_end || transpose_cursor < transpose_end) {
    if (transpose_cursor >= transpose_end
        || (row_cursor < row_end && matrix.col_ind[row_cursor] <= transpose_rows[transpose_cursor]))
      neighbour = matrix.col_ind[row_cursor++];
    else
      neighbour = transpose_rows[transpose_cursor++];
    if (neighbour == row || neighbour == previous) continue;
    previous = neighbour;
    if (!count_only) output[written] = neighbour;
    ++written;
  }
  return written;
}

inline SymmetricPattern symmetric_pattern_of(const CsrMatrix& matrix) {
  SymmetricPattern     pattern;
  std::vector<int64_t> transpose_ptr;
  std::vector<int>     transpose_rows;

  if (matrix.rows != matrix.cols) throw std::runtime_error("RCM requires a square matrix");
  transpose_pattern(matrix, transpose_ptr, transpose_rows);
  pattern.adjacency_ptr.assign((size_t)matrix.rows + 1, 0);
  for (int row = 0; row < matrix.rows; ++row)
    pattern.adjacency_ptr[row + 1] = pattern.adjacency_ptr[row]
                                   + merge_row_neighbours(matrix, transpose_ptr, transpose_rows, row, true, nullptr);
  pattern.adjacency.resize((size_t)pattern.adjacency_ptr[matrix.rows]);
  for (int row = 0; row < matrix.rows; ++row)
    merge_row_neighbours(matrix, transpose_ptr, transpose_rows, row, false,
                         pattern.adjacency.data() + pattern.adjacency_ptr[row]);
  return pattern;
}

inline int64_t degree_of(const SymmetricPattern& pattern, int node) {
  return pattern.adjacency_ptr[node + 1] - pattern.adjacency_ptr[node];
}

// Breadth-first level structure from root, restricted to one connected component.
// distance[] must be -1 on every node of the component on entry and is left filled;
// component_nodes receives the nodes in visiting order, last_level those at the
// largest distance. Returns the eccentricity of root.
inline int level_structure(const SymmetricPattern& pattern, int root, std::vector<int>& distance,
                           std::vector<int>& component_nodes, std::vector<int>& last_level) {
  size_t head          = 0;
  int    node          = 0;
  int    neighbour     = 0;
  int    eccentricity  = 0;

  component_nodes.clear();
  last_level.clear();
  distance[root] = 0;
  component_nodes.push_back(root);
  while (head < component_nodes.size()) {
    node = component_nodes[head++];
    for (int64_t slot = pattern.adjacency_ptr[node]; slot < pattern.adjacency_ptr[node + 1]; ++slot) {
      neighbour = pattern.adjacency[slot];
      if (distance[neighbour] >= 0) continue;
      distance[neighbour] = distance[node] + 1;
      component_nodes.push_back(neighbour);
    }
  }
  eccentricity = distance[component_nodes.back()];
  for (size_t position = component_nodes.size(); position-- > 0;) {
    if (distance[component_nodes[position]] != eccentricity) break;
    last_level.push_back(component_nodes[position]);
  }
  return eccentricity;
}

inline void reset_distances(const std::vector<int>& component_nodes, std::vector<int>& distance) {
  for (int node : component_nodes) distance[node] = -1;
}

// George-Liu pseudo-peripheral node: repeatedly restart the level structure from the
// lowest-degree node of the last level while the eccentricity keeps growing.
inline int pseudo_peripheral_node(const SymmetricPattern& pattern, int seed, std::vector<int>& distance,
                                  std::vector<int>& component_nodes) {
  std::vector<int> last_level;
  std::vector<int> candidate_nodes;
  std::vector<int> candidate_last_level;
  int              current              = seed;
  int              current_eccentricity = 0;
  int              candidate            = 0;
  int              candidate_eccentricity = 0;

  current_eccentricity = level_structure(pattern, current, distance, component_nodes, last_level);
  for (int iteration = 0; iteration < PSEUDO_PERIPHERAL_ITERATIONS; ++iteration) {
    candidate = last_level.front();
    for (int node : last_level)
      if (degree_of(pattern, node) < degree_of(pattern, candidate)) candidate = node;
    reset_distances(component_nodes, distance);
    candidate_eccentricity = level_structure(pattern, candidate, distance, candidate_nodes, candidate_last_level);
    if (candidate_eccentricity <= current_eccentricity) break;
    current              = candidate;
    current_eccentricity = candidate_eccentricity;
    component_nodes.swap(candidate_nodes);
    last_level.swap(candidate_last_level);
  }
  return current;
}

// Cuthill-McKee from start: breadth first, each node's unvisited neighbours appended
// in increasing degree (ties by index, so the ordering is deterministic).
inline void cuthill_mckee_component(const SymmetricPattern& pattern, int start, std::vector<char>& visited,
                                    std::vector<int>& order) {
  std::vector<int> fresh_neighbours;
  size_t           head      = order.size();
  int              node      = 0;
  int              neighbour = 0;

  visited[start] = 1;
  order.push_back(start);
  while (head < order.size()) {
    node = order[head++];
    fresh_neighbours.clear();
    for (int64_t slot = pattern.adjacency_ptr[node]; slot < pattern.adjacency_ptr[node + 1]; ++slot) {
      neighbour = pattern.adjacency[slot];
      if (visited[neighbour]) continue;
      visited[neighbour] = 1;
      fresh_neighbours.push_back(neighbour);
    }
    std::sort(fresh_neighbours.begin(), fresh_neighbours.end(), [&](int left, int right) {
      const int64_t left_degree  = degree_of(pattern, left);
      const int64_t right_degree = degree_of(pattern, right);
      return left_degree != right_degree ? left_degree < right_degree : left < right;
    });
    order.insert(order.end(), fresh_neighbours.begin(), fresh_neighbours.end());
  }
}

// Returns new_to_old. Components are ordered by their lowest-index node.
inline std::vector<int> reverse_cuthill_mckee(const CsrMatrix& matrix) {
  const SymmetricPattern pattern = symmetric_pattern_of(matrix);
  std::vector<int>       distance((size_t)matrix.rows, -1);
  std::vector<char>      visited((size_t)matrix.rows, 0);
  std::vector<int>       order;
  std::vector<int>       component_nodes;
  int                    start = 0;

  order.reserve((size_t)matrix.rows);
  for (int seed = 0; seed < matrix.rows; ++seed) {
    if (visited[seed]) continue;
    start = pseudo_peripheral_node(pattern, seed, distance, component_nodes);
    reset_distances(component_nodes, distance);
    cuthill_mckee_component(pattern, start, visited, order);
  }
  std::reverse(order.begin(), order.end());
  return order;
}

// B = P A P^T with b_ij = a_{new_to_old[i], new_to_old[j]}; columns sorted within rows.
inline CsrMatrix apply_symmetric_permutation(const CsrMatrix& source, const std::vector<int>& new_to_old) {
  CsrMatrix        result;
  std::vector<int> old_to_new((size_t)source.rows, 0);
  int              old_row     = 0;
  int              destination = 0;

  for (int new_index = 0; new_index < source.rows; ++new_index) old_to_new[new_to_old[new_index]] = new_index;
  result.rows               = source.rows;
  result.cols               = source.cols;
  result.symmetric_expanded = source.symmetric_expanded;
  result.name               = source.name;
  result.row_ptr.assign((size_t)source.rows + 1, 0);
  for (int new_row = 0; new_row < source.rows; ++new_row) {
    old_row = new_to_old[new_row];
    result.row_ptr[new_row + 1] = result.row_ptr[new_row] + (source.row_ptr[old_row + 1] - source.row_ptr[old_row]);
  }
  result.col_ind.resize(source.col_ind.size());
  result.val.resize(source.val.size());
  for (int new_row = 0; new_row < source.rows; ++new_row) {
    old_row     = new_to_old[new_row];
    destination = result.row_ptr[new_row];
    for (int entry = source.row_ptr[old_row]; entry < source.row_ptr[old_row + 1]; ++entry, ++destination) {
      result.col_ind[destination] = old_to_new[source.col_ind[entry]];
      result.val[destination]     = source.val[entry];
    }
  }
  sort_columns_within_rows(result);
  result.compute_stats();
  return result;
}

// "<dir>/<name>.mtx" -> "<dir>/<name>.<ordering>.perm"
inline std::string permutation_path_for(const std::string& matrix_path, const std::string& ordering) {
  const size_t extension = matrix_path.rfind(".mtx");
  const std::string stem = (extension == std::string::npos) ? matrix_path : matrix_path.substr(0, extension);

  return stem + "." + ordering + ".perm";
}

inline void write_permutation(const std::string& path, const std::vector<int>& new_to_old, double seconds) {
  const int64_t rows          = (int64_t)new_to_old.size();
  const std::string temporary = path + ".partial";
  std::FILE*    output        = std::fopen(temporary.c_str(), "wb");

  if (!output) throw std::runtime_error("cannot write permutation " + temporary);
  std::fwrite(PERMUTATION_MAGIC, 1, sizeof(PERMUTATION_MAGIC), output);
  std::fwrite(&rows, sizeof(rows), 1, output);
  std::fwrite(&seconds, sizeof(seconds), 1, output);
  std::fwrite(new_to_old.data(), sizeof(int), new_to_old.size(), output);
  std::fclose(output);
  std::rename(temporary.c_str(), path.c_str());
}

// Returns false if the file does not exist; throws if it exists but is not a valid
// permutation of expected_rows entries.
inline bool read_permutation(const std::string& path, int expected_rows, std::vector<int>& new_to_old,
                             double& seconds) {
  std::FILE*        input = std::fopen(path.c_str(), "rb");
  char              magic[8];
  int64_t           rows  = 0;
  std::vector<char> seen;
  bool              valid = true;

  if (!input) return false;
  valid = std::fread(magic, 1, sizeof(magic), input) == sizeof(magic)
       && std::equal(magic, magic + sizeof(magic), PERMUTATION_MAGIC)
       && std::fread(&rows, sizeof(rows), 1, input) == 1 && rows == expected_rows
       && std::fread(&seconds, sizeof(seconds), 1, input) == 1;
  if (valid) {
    new_to_old.resize((size_t)rows);
    valid = std::fread(new_to_old.data(), sizeof(int), new_to_old.size(), input) == new_to_old.size();
  }
  std::fclose(input);
  if (valid) {
    seen.assign((size_t)rows, 0);
    for (int old_index : new_to_old) {
      if (old_index < 0 || old_index >= expected_rows || seen[old_index]) { valid = false; break; }
      seen[old_index] = 1;
    }
  }
  if (!valid) throw std::runtime_error("invalid permutation file " + path);
  return true;
}

// Loads the RCM permutation saved next to the matrix, computing and saving it when
// absent, and returns the reordered matrix. seconds is the time the ordering took
// when it was computed (stored in the file, so it survives later loads).
inline CsrMatrix reorder_with_rcm(const std::string& matrix_path, const CsrMatrix& matrix, double& seconds) {
  const std::string permutation_path = permutation_path_for(matrix_path, "rcm");
  std::vector<int>  new_to_old;
  std::chrono::steady_clock::time_point start_time;

  if (!read_permutation(permutation_path, matrix.rows, new_to_old, seconds)) {
    start_time = std::chrono::steady_clock::now();
    new_to_old = reverse_cuthill_mckee(matrix);
    seconds    = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    write_permutation(permutation_path, new_to_old, seconds);
  }
  return apply_symmetric_permutation(matrix, new_to_old);
}
