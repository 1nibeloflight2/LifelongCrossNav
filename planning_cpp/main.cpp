#include <random>

#include <iostream>
#include <queue>
#include <vector>
#include <limits>
#include <chrono>
#include <rerun.hpp>

using namespace std;
using namespace std;
using namespace std::chrono;
const int INF = numeric_limits<int>::max();

struct Point {
    int x, y;
};

struct Compare {
    bool operator()(const pair<int, Point>& a, const pair<int, Point>& b) {
        return a.first > b.first;
    }
};

void generateObstacles(vector<vector<int>>& grid, int numObstacles) {
    random_device rd;
    mt19937 gen(rd());
    int centerX = grid[0].size() / 2;
    int centerY = grid.size() / 2;
    int deviationX = grid[0].size() / 4;
    int deviationY = grid.size() / 4;

    normal_distribution<> xDist(centerX, deviationX);
    normal_distribution<> yDist(centerY, deviationY);
    uniform_int_distribution<int> widthDist(1, 50);
    uniform_int_distribution<int> heightDist(1, 100);

    for (int i = 0; i < numObstacles; i++) {
        int x = round(xDist(gen));
        int y = round(yDist(gen));
        int width = widthDist(gen);
        int height = heightDist(gen);

        // Ensure the obstacle doesn't go out of bounds
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x + width > grid[0].size()) {
            width = grid[0].size() - x;
        }
        if (y + height > grid.size()) {
            height = grid.size() - y;
        }

        for (int j = y; j < y + height; j++) {
            for (int k = x; k < x + width; k++) {
                grid[j][k] = 0; // Mark the cell as an obstacle
            }
        }
    }
}
int distanceTransform(vector<vector<int>>& grid) {
    int rows = grid.size();
    int cols = grid[0].size();
    vector<vector<int>> dt(rows, vector<int>(cols, INF));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (grid[i][j] == 0) {
                dt[i][j] = INF;
            } else {
                dt[i][j] = 1;
            }
        }
    }

    for (int _ = 0; _ < 2; _++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 0) {
                    continue;
                }
                int minDist = INF;
                if (i > 0) {
                    minDist = min(minDist, dt[i-1][j]);
                }
                if (i < rows - 1) {
                    minDist = min(minDist, dt[i+1][j]);
                }
                if (j > 0) {
                    minDist = min(minDist, dt[i][j-1]);
                }
                if (j < cols - 1) {
                    minDist = min(minDist, dt[i][j+1]);
                }
                dt[i][j] = 1 + minDist;
            }
        }
    }

    return 0;
}

std::vector<std::vector<Point>> dijkstra(vector<vector<int>>& grid, Point start, vector<Point>& goalPoints) {
    std::vector<std::vector<Point>> paths;
    for (int i = 0; i < goalPoints.size(); i++) {
        paths.emplace_back();
    }
    int rows = grid.size();
    int cols = grid[0].size();
    vector<vector<int>> distances(rows, vector<int>(cols, INF));
    vector<vector<Point>> previous(rows, vector<Point>(cols, {-1, -1}));
    distances[start.x][start.y] = 0;
    priority_queue<pair<int, Point>, vector<pair<int, Point>>, Compare> queue;
    queue.push({0, start});

    while (!queue.empty()) {
        pair<int, Point> current = queue.top();
        queue.pop();
        int currentDist = current.first;
        Point currentPos = current.second;

        // Skip if the current distance is not up-to-date
        if (currentDist > distances[currentPos.x][currentPos.y]) {
            continue;
        }

        // Check all four cardinal directions
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                if (abs(dx) + abs(dy) == 1) {
                    int x = currentPos.x + dx;
                    int y = currentPos.y + dy;
                    if (x < 0 || x >= rows || y < 0 || y >= cols || grid[x][y] == 0) {
                        continue; // Skip out of bounds or obstacles
                    }
                    int newDist = currentDist + 1; // Uniform cost for moving
                    if (newDist < distances[x][y]) {
                        distances[x][y] = newDist;
                        queue.push({newDist, {x, y}});
                        previous[x][y] = currentPos;

                    }
                }
            }
        }
    }
    for (int i = 0; i < goalPoints.size(); i++) {
        int goalX = goalPoints[i].x;
        int goalY = goalPoints[i].y;
        int distance = distances[goalX][goalY];

        if (distance == INF) {
        } else {
            for (Point at = goalPoints[i]; at.x != -1; at = previous[at.x][at.y]) {
                paths[i].emplace_back(at);
            }
        }
    }
}

int main() {
    const auto rec = rerun::RecordingStream("rerun_example_cpp");
    rec.spawn().exit_on_failure();

    int rows = 1000;
    int cols = 1000;
    vector<vector<int>> grid(rows, vector<int>(cols, 1)); // Initialize grid with all 1s (no obstacles)

    int numObstacles = 200;
    generateObstacles(grid, numObstacles);

    Point start = {0, 0};
    vector<Point> goalPoints = {{100, 100}, {800, 800}, {300, 300}};
    vector<int> goalScores = {11, 5, 8};
    auto start_time = high_resolution_clock::now();
    auto paths = dijkstra(grid, start, goalPoints);
    // Get ending timepoint
    auto stop = high_resolution_clock::now();

    // Get duration. Substart timepoints to
    // get duration. To cast it to proper unit
    // use duration cast method
    auto duration = duration_cast<microseconds>(stop - start_time);

    cout << "Time taken by function: "
         << duration.count()/1000.0/1000.0 << " seconds" << endl;
    return 0;
}
