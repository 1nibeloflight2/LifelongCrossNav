#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include "matrix_utils.h"
#include <iostream>

namespace py = pybind11;

const float INF = std::numeric_limits<float>::max();

struct Point {
    int x, y;
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

template <typename T> struct Compare {
    bool operator()(const std::pair<T, Point>& a, const std::pair<T, Point>& b) {
        return a.first > b.first;
    }
};

template <typename MatrixType>
Eigen::ArrayXXi inverseDilate(const Eigen::Map<const MatrixType>& input, int kernelSize) {
    int rows = input.rows();
    int cols = input.cols();
    int radius = kernelSize / 2;

    Eigen::ArrayXXi output = Eigen::ArrayXXi::Ones(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (input(i, j) == 0) {
                int startRow = std::max(0, i - radius);
                int endRow = std::min(rows - 1, i + radius);
                int startCol = std::max(0, j - radius);
                int endCol = std::min(cols - 1, j + radius);

                output.block(startRow, startCol, endRow - startRow + 1, endCol - startCol + 1).setZero();
            }
        }
    }

    return output;
}

template <typename MatrixType>
Eigen::ArrayXXf gradientInverseDilate(const Eigen::Map<const MatrixType>& input, int kernelSize) {
    int rows = input.rows();
    int cols = input.cols();
    int radius = kernelSize / 2;
    Eigen::ArrayXXf output = Eigen::ArrayXXf::Ones(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (input(i, j) == 0) {
                int startRow = std::max(0, i - radius);
                int endRow = std::min(rows - 1, i + radius);
                int startCol = std::max(0, j - radius);
                int endCol = std::min(cols - 1, j + radius);

                for (int r = startRow; r <= endRow; ++r) {
                    for (int c = startCol; c <= endCol; ++c) {
                        float distance = std::sqrt(std::pow(r - i, 2) + std::pow(c - j, 2));
                        float value = std::min(distance / radius, 1.0f);
                        output(r, c) = std::min(output(r, c), value);
                    }
                }
            }
        }
    }

    return output;
}

template <typename MaskType, typename ScoreType>
std::tuple<float, int, std::pair<int, int>, Eigen::MatrixXf> compute_frontier_scores_impl(
    std::pair<int, int> start,
    const Eigen::Map<const MaskType>& mask_coverage,
    const Eigen::Map<const ScoreType>& scores,
    int max_depth
) {
    int height = mask_coverage.rows();
    int width = mask_coverage.cols();
    Point start_point = {start.first, start.second};

    Eigen::MatrixXf reachable = Eigen::MatrixXf::Zero(height, width);
    Eigen::MatrixXf depth_map = Eigen::MatrixXf::Constant(height, width, (float)max_depth + 1.0f);

    std::priority_queue<std::pair<float, Point>, std::vector<std::pair<float, Point>>, Compare<float>> queue;
    queue.push(std::make_pair(0.0f, start_point));

    float max_score = 0.0f;
    int num_visited = 0;
    std::pair<int, int> max_pos = {0, 0};
    const std::array<std::pair<int, int>, 4> directions = {{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};

    while (!queue.empty()) {
        auto [depth, currentPos] = queue.top();
        queue.pop();

        if (depth >= max_depth || depth >= depth_map(currentPos.x, currentPos.y) || mask_coverage(currentPos.x, currentPos.y) == 0) {
            continue;
        }
        depth_map(currentPos.x, currentPos.y) = depth;
        num_visited++;
        float current_score = scores(currentPos.x, currentPos.y);
        reachable(currentPos.x, currentPos.y) = current_score;
        if (current_score > max_score) {
            max_score = current_score;
            max_pos = {currentPos.x, currentPos.y};
        }

        for (const auto& [dx, dy] : directions) {
            int nx = currentPos.x + dx, ny = currentPos.y + dy;
            if (nx >= 0 && nx < height && ny >= 0 && ny < width) {
                float new_depth = depth + 1;
                if (new_depth < depth_map(nx, ny) && mask_coverage(nx, ny) > 0) {
                    queue.push({static_cast<float>(new_depth), {nx, ny}});
                }
            }
        }
    }

    return std::make_tuple(max_score, num_visited, max_pos, reachable);
}

template <typename MatrixType>
std::vector<std::pair<float, std::vector<std::pair<int, int>>>> dijkstra(
    const Eigen::Map<const MatrixType>& grid,
    std::pair<int, int> start,
    std::vector<std::pair<int, int>> goalPoints,
    int obstcl_kernel_size
    ) {

    int rows = grid.rows();
    int cols = grid.cols();

    Point start_point = {start.first, start.second};
    std::vector<Point> goal_points;
    for (const auto& gp : goalPoints) {
        goal_points.push_back({gp.first, gp.second});
    }

    Eigen::MatrixXf distances = Eigen::MatrixXf::Constant(rows, cols, INF);
    std::vector<std::vector<Point>> previous(rows, std::vector<Point>(cols, {-1, -1}));
    distances(start_point.x, start_point.y) = 0.0f;

    std::priority_queue<std::pair<float, Point>, std::vector<std::pair<float, Point>>, Compare<float>> queue;
    queue.push({0, start_point});
    // TODO Make gussian blurring instead of dilating, or make it a parameter
    auto dilated = gradientInverseDilate(grid, obstcl_kernel_size);
    while (!queue.empty()) {
        auto [currentDist, currentPos] = queue.top();
        queue.pop();

        if (currentDist > distances(currentPos.x, currentPos.y)) {
            continue;
        }

        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (abs(dx) + abs(dy) > 0) {
                    int x = currentPos.x + dx;
                    int y = currentPos.y + dy;
                    if (x < 0 || x >= rows || y < 0 || y >= cols || grid(x, y) == 0) {
                        continue;
                    }
                    float newDist = currentDist + std::sqrt(abs(dx) + abs(dy));
                    if (dilated(x, y) < 1) {
                        newDist += 2.0f * (1.0 - dilated(x, y));
                    }
                    if (newDist < distances(x, y)) {
                        distances(x, y) = newDist;
                        queue.push({newDist, {x, y}});
                        previous[x][y] = currentPos;
                    }
                }
            }
        }
    }

    std::vector<std::pair<float, std::vector<std::pair<int, int>>>> paths(goal_points.size());
    for (size_t i = 0; i < goal_points.size(); i++) {
        Point at = goal_points[i];
        if (distances(at.x, at.y) != INF) {
            paths[i].first = distances(at.x, at. y);
            while (!(at == start_point)) {
                paths[i].second.emplace_back(at.x, at.y);
                at = previous[at.x][at.y];
            }
            paths[i].second.emplace_back(start_point.x, start_point.y);
            std::reverse(paths[i].second.begin(), paths[i].second.end());
        }
    }
    return paths;
}

template <typename MatrixType, typename FeasType>
std::vector<std::pair<int, int>> a_star_range(
    const Eigen::Map<const MatrixType>& grid,
    const Eigen::Map<const FeasType>& feasible_goal_pts,
    std::pair<int, int> start,
    std::pair<int, int> goalPoint,
    int obstcl_kernel_size,
    int goal_range
) {

    int rows = grid.rows();
    int cols = grid.cols();

    Point start_point = {start.first, start.second};
    Point goal_point = {goalPoint.first, goalPoint.second};

    Eigen::MatrixXf heuristic = Eigen::MatrixXf::Constant(rows, cols, INF);
    std::vector<std::vector<Point>> previous(rows, std::vector<Point>(cols, {-1, -1}));
    heuristic(start_point.x, start_point.y) = 0.0f;

    std::priority_queue<std::pair<float, Point>, std::vector<std::pair<float, Point>>, Compare<float>> queue;
    queue.push({0, start_point});
    // TODO Make this kernel size a parameter
    auto dilated = inverseDilate(grid, obstcl_kernel_size);
    while (!queue.empty()) {
        auto [currentDist, currentPos] = queue.top();
        queue.pop();

        if (currentDist > heuristic(currentPos.x, currentPos.y)) {
            continue;
        }
        if (feasible_goal_pts(currentPos.x, currentPos.y) &&
                std::sqrt((currentPos.x - goal_point.x) * (currentPos.x - goal_point.x) + (currentPos.y - goal_point.y)
            * (currentPos.y - goal_point.y)) <= goal_range) {
            std::vector<std::pair<int, int>> path;
            Point at = currentPos;
            if (heuristic(at.x, at.y) != INF) {
                while (!(at == start_point)) {
                    path.emplace_back(at.x, at.y);
                    at = previous[at.x][at.y];
                }
                path.emplace_back(start_point.x, start_point.y);
                std::reverse(path.begin(), path.end());
                return path;
            }
        }
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (abs(dx) + abs(dy) > 0) {
                    int x = currentPos.x + dx;
                    int y = currentPos.y + dy;
                    if (x < 0 || x >= rows || y < 0 || y >= cols || grid(x, y) == 0) {
                        continue;
                    }
                    float newHeur = currentDist + std::sqrt(abs(dx) + abs(dy)) + std::sqrt((x - goal_point.x) * (x - goal_point.x) + (y - goal_point.y) * (y - goal_point.y));
                    if (dilated(x, y) == 0) {
                        newHeur += 10.0f;
                    }
                    if (newHeur < heuristic(x, y)) {
                        heuristic(x, y) = newHeur;
                        queue.push({newHeur, {x, y}});
                        previous[x][y] = currentPos;
                    }
                }
            }
        }
    }
    return {};
}

std::tuple<float, int, std::pair<int, int>, Eigen::MatrixXf> compute_frontier_scores_wrapper(
    std::pair<int, int> start,
    py::array& mask_coverage_,
    py::array& scores_,
    int max_depth
) {
    auto mask_coverage = get_matrix(mask_coverage_);
    auto scores = get_matrix(scores_);
    return std::visit([&](const auto& mask, const auto& score) {
      return compute_frontier_scores_impl(start, mask, score, max_depth);
    }, mask_coverage, scores);
}

template<typename MaskType>
std::vector<std::pair<float, std::vector<std::pair<int, int>>>> dijkstra_wrapper_impl(
    const Eigen::Map<const MaskType>& grid,
    std::pair<int, int> start,
    std::vector<std::pair<int, int>> goalPoints,
    int obstcl_kernel_size = 0
) {
    return dijkstra(grid, start, goalPoints, obstcl_kernel_size);
}

std::vector<std::pair<float, std::vector<std::pair<int, int>>>> dijkstra_wrapper(
    py::array& grid_,
    std::pair<int, int> start,
    std::vector<std::pair<int, int>> goalPoints,
    int obstcl_kernel_size
) {
    auto grid = get_matrix(grid_);
    return std::visit([&](const auto& grid) {
      return dijkstra_wrapper_impl(grid, start, goalPoints, obstcl_kernel_size);
    }, grid);
}

std::vector<std::pair<int, int>> a_star_range_wrapper(
    py::array& grid_,
    py::array& feasible_goal_pts_,
    std::pair<int, int> start,
    std::pair<int, int> goalPoint,
    int obstcl_kernel_size,
    int goal_range
) {
    auto grid = get_matrix(grid_);
    auto feasible_goal_pts = get_matrix(feasible_goal_pts_);
    return std::visit([&](const auto& grid, const auto& feasible_goal_pts) {
      return a_star_range(grid, feasible_goal_pts, start, goalPoint, obstcl_kernel_size, goal_range);
    }, grid, feasible_goal_pts);
}

PYBIND11_MODULE(planning_utils_cpp, m) {
    m.doc() = "XX";
    m.def("dijkstra", &dijkstra_wrapper, "A function that performs Dijkstra's algorithm",
          py::arg("grid").noconvert(), py::arg("start"), py::arg("goal_points"), py::arg("obstcl_kernel_size"));
    m.def("compute_reachable_area", &compute_frontier_scores_wrapper,
          "Compute the score and number of visited nodes for a given start position (optimized version)",
          py::arg("start"), py::arg("mask_coverage").noconvert(), py::arg("scores").noconvert(), py::arg("max_depth"));
    m.def("a_star_range", &a_star_range_wrapper, "A function that performs A* search with a goal range",
            py::arg("grid").noconvert(), py::arg("feasible_goal_pts").noconvert(), py::arg("start"), py::arg("goal_point"), py::arg("obstcl_kernel_size"), py::arg("goal_range"));
}
