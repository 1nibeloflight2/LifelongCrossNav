import numpy as np
import time
from planning_utils_cpp import dijkstra, compute_reachable_area, a_star_range  # Import our custom Dijkstra module
import matplotlib.pyplot as plt
from py_impl import compute_reachable_area as py_func

def generate_obstacles(grid, num_obstacles):
    rows, cols = grid.shape
    center_x, center_y = rows // 2, cols // 2
    deviation_x, deviation_y = rows // 4, cols // 4

    for _ in range(num_obstacles):
        x = int(np.random.normal(center_x, deviation_x))
        y = int(np.random.normal(center_y, deviation_y))
        width = np.random.randint(1, 51)
        height = np.random.randint(1, 101)

        x = max(0, min(x, rows - 1))
        y = max(0, min(y, cols - 1))
        width = min(width, rows - x)
        height = min(height, cols - y)

        grid[x:x+width, y:y+height] = 0  # Note the swapped x and y due to Fortran-order

def plot_grid_and_paths(grid, start, goal_points, paths, scores):
    plt.figure(figsize=(12, 12))
    print(grid.shape)
    grid = grid.T
    # grid = np.flip(grid, axis=0)
    plt.imshow(grid, cmap='binary')  # Transpose the grid for correct orientation

    # Plot start point
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')

    # Plot goal points
    for i, goal in enumerate(goal_points):
        plt.plot(goal[0], goal[1], 'r*', markersize=10, label=f'Goal {i+1}' if i == 0 else "")

    # Plot paths
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i, path_and_score in enumerate(paths):
        score, path = path_and_score
        if path:
            path_array = np.array(path)
            plt.plot(path_array[:, 0], path_array[:, 1], color=colors[i % len(colors)],
                     linewidth=2, label=f'Path {i+1}')

    plt.legend()
    plt.title("Dijkstra's Algorithm: Obstacles and Paths")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def visualize_results(mask, scores, reachable, title):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(mask, cmap='binary')
    ax1.set_title('Mask')
    ax1.axis('off')

    ax2.imshow(scores, cmap='viridis')
    ax2.set_title('Scores')
    ax2.axis('off')

    ax3.imshow(reachable, cmap='hot')
    ax3.set_title('Reachable Area')
    ax3.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    rows, cols = 1000, 1200
    # Create a Fortran-contiguous array
    grid = np.ones((rows, cols), dtype=np.int64, order='F')
    num_obstacles = 500
    generate_obstacles(grid, num_obstacles)

    start = (1, 1)
    # generate N random goal points
    N = 1
    goal_points = [(np.random.randint(0, rows), np.random.randint(0, cols)) for _ in range(N)]
    scores = [(np.random.randint(1, 100)) for _ in range(N)]
    path_scores = []
    start_time = time.time()
    paths = dijkstra(grid, start, goal_points, 30)
    paths = [(1, a_star_range(grid, start, goal_points[0], 30, 100))]
    print(grid[0,0])
    end_time = time.time()

    print(f"Time taken by function: {end_time - start_time:.6f} seconds")

    # Print out the paths
    for i, path_ in enumerate(paths):
        s, path = path_
        print(f"Path to goal {goal_points[i]}:")
        if path:
            print(f"Score to goal {s / scores[i]**2}:")
            path_scores.append(scores[i]**2./ s )
            print(f"  Length: {s}")
            print(f"  Start: {path[0]}")
            print(f"  End: {path[-1]}")
        else:
            path_scores.append(1)
            print("  No path found")
    path_scores = np.array(path_scores)
    path_scores = (path_scores - path_scores.min()) / (path_scores.max() - path_scores.min())
    print(path_scores.min(), path_scores.max())
    # Plot the grid, obstacles, and paths
    plot_grid_and_paths(grid, start, goal_points, paths, path_scores)


    # Path planning end, now compute reachable area
    # we need a grid sized score array with the scores of the map, we define a sine here
    x = np.linspace(0, 2 * np.pi, cols)
    y = np.linspace(0, 2 * np.pi, rows)
    xx, yy = np.meshgrid(x, y)
    scores = np.sin(xx) + np.cos(yy)
    scores = np.asfortranarray(scores)
    grid = grid.astype(np.bool)
    grid = np.asfortranarray(grid)
    py_score, py_visited, py_reachable = py_func(np.array(start), grid, scores * (-1), 500)

    py_start_time = time.time()
    py_score, py_visited, py_reachable = py_func(np.array(start), grid, scores, 500)
    py_end_time = time.time()
    py_execution_time = py_end_time - py_start_time

    # Run and time C++ implementation
    cpp_start_time = time.time()
    cpp_score, cpp_visited, cpp_reachable = compute_reachable_area(start, grid, scores, 500)
    cpp_end_time = time.time()
    cpp_execution_time = cpp_end_time - cpp_start_time
    print(f"Python Execution Time: {py_execution_time:.6f} seconds")
    print(f"C++ Execution Time: {cpp_execution_time:.6f} seconds")
    print(f"Speedup: {py_execution_time / cpp_execution_time:.2f}x")
    visualize_results(grid, scores, py_reachable, "Python Implementation")
    visualize_results(grid, scores, cpp_reachable, "C++ Implementation")

if __name__ == "__main__":
    main()