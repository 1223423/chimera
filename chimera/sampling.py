import numpy as np
import itertools
import math


def generate_gaussian(n, width, height):
    """
    Gaussian position generator for an interesting centralized effect

    Args:
        n: Number of positions to generate
        width: Canvas width
        height: Canvas height

    Returns:
        List of (x, y) tuples
    """
    positions = set()
    center_x, center_y = width / 2, height / 2
    std_dev_x, std_dev_y = width / 6, height / 6

    while len(positions) < n:
        x = int(np.random.normal(center_x, std_dev_x))
        y = int(np.random.normal(center_y, std_dev_y))

        if 0 <= x < width and 0 <= y < height:
            positions.add((x, y))

    return list(positions)


def generate_grid(width, height, increment_width, increment_height):
    """
    Grid generator; mostly used to oversample before applying uniform random positions

    Args:
        width: Canvas width
        height: Canvas height
        increment_width: Horizontal spacing
        increment_height: Vertical spacing

    Returns:
        List of (x, y) tuples
    """
    return list(itertools.product(
        range(0, width, int(increment_width)),
        range(0, height, int(increment_height))
    ))


def generate_uniform(n, width, height):
    """
    Uniform random generator; mostly used for adding random over-sampled jitter aesthetic

    Args:
        n: Number of positions to generate
        width: Canvas width
        height: Canvas height

    Returns:
        List of (x, y) tuples
    """
    return list(zip(
        np.random.randint(0, width, n),
        np.random.randint(0, height, n)
    ))


def generate_poisson_disk(n, width, height, min_distance=None):
    """
    Poisson disk sampling for evenly distributed but random-looking positions
    Avoids clustering while maintaining organic appearance

    Args:
        n: Number of positions to generate
        width: Canvas width
        height: Canvas height
        min_distance: Minimum distance between points (auto-calculated if None)

    Returns:
        List of (x, y) tuples
    """
    if min_distance is None:
        # Calculate min distance based on desired number of samples
        area = width * height
        min_distance = math.sqrt(area / n) * 0.7

    positions = []
    active_list = []
    grid_size = int(min_distance / math.sqrt(2))
    grid = {}

    # Start with random initial point
    initial = (np.random.randint(0, width), np.random.randint(0, height))
    positions.append(initial)
    active_list.append(initial)
    grid[(initial[0] // grid_size, initial[1] // grid_size)] = initial

    attempts_per_point = 30

    while active_list and len(positions) < n:
        idx = np.random.randint(0, len(active_list))
        point = active_list[idx]
        found = False

        for _ in range(attempts_per_point):
            angle = np.random.uniform(0, 2 * math.pi)
            radius = np.random.uniform(min_distance, 2 * min_distance)
            new_x = int(point[0] + radius * math.cos(angle))
            new_y = int(point[1] + radius * math.sin(angle))

            if 0 <= new_x < width and 0 <= new_y < height:
                grid_x, grid_y = new_x // grid_size, new_y // grid_size
                valid = True

                # Check neighboring grid cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_key = (grid_x + dx, grid_y + dy)
                        if neighbor_key in grid:
                            neighbor = grid[neighbor_key]
                            dist = math.sqrt((new_x - neighbor[0])**2 + (new_y - neighbor[1])**2)
                            if dist < min_distance:
                                valid = False
                                break
                    if not valid:
                        break

                if valid:
                    new_point = (new_x, new_y)
                    positions.append(new_point)
                    active_list.append(new_point)
                    grid[(grid_x, grid_y)] = new_point
                    found = True
                    break

        if not found:
            active_list.pop(idx)

    # If we didn't get enough points, fill with random positions
    while len(positions) < n:
        positions.append((np.random.randint(0, width), np.random.randint(0, height)))

    return positions[:n]


def generate_spiral(n, width, height):
    """
    Spiral pattern from center outward - creates interesting vortex effect

    Args:
        n: Number of positions to generate
        width: Canvas width
        height: Canvas height

    Returns:
        List of (x, y) tuples
    """
    positions = []
    center_x, center_y = width / 2, height / 2
    max_radius = math.sqrt((width/2)**2 + (height/2)**2)

    for i in range(n):
        progress = i / n
        radius = progress * max_radius
        angle = progress * 8 * math.pi  # 4 full rotations

        x = int(center_x + radius * math.cos(angle))
        y = int(center_y + radius * math.sin(angle))

        # Add small jitter for organic feel
        x += np.random.randint(-2, 3)
        y += np.random.randint(-2, 3)

        if 0 <= x < width and 0 <= y < height:
            positions.append((x, y))

    return positions


def generate_concentric(n, width, height):
    """
    Concentric circles from center outward

    Args:
        n: Number of positions to generate
        width: Canvas width
        height: Canvas height

    Returns:
        List of (x, y) tuples
    """
    positions = []
    center_x, center_y = width / 2, height / 2
    max_radius = math.sqrt((width/2)**2 + (height/2)**2)

    n_circles = int(math.sqrt(n))
    points_per_circle = n // n_circles

    for circle in range(n_circles):
        radius = (circle + 1) * max_radius / n_circles
        n_points = points_per_circle if circle < n_circles - 1 else n - len(positions)

        for i in range(n_points):
            angle = (i / n_points) * 2 * math.pi
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))

            if 0 <= x < width and 0 <= y < height:
                positions.append((x, y))

    return positions


def generate_halton(n, width, height):
    """
    Halton sequence - low-discrepancy quasi-random sampling
    Better coverage than pure random

    Args:
        n: Number of positions to generate
        width: Canvas width
        height: Canvas height

    Returns:
        List of (x, y) tuples
    """
    def halton_sequence(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f = f / base
            result = result + f * (i % base)
            i = i // base
        return result

    positions = []
    for i in range(n):
        x = int(halton_sequence(i, 2) * width)
        y = int(halton_sequence(i, 3) * height)
        if 0 <= x < width and 0 <= y < height:
            positions.append((x, y))

    return positions


def generate_jittered_grid(n, width, height, jitter=0.5):
    """
    Regular grid with random jitter - balances coverage and randomness

    Args:
        n: Number of positions to generate
        width: Canvas width
        height: Canvas height
        jitter: Jitter amount (0.0-1.0, where 1.0 = full cell size)

    Returns:
        List of (x, y) tuples
    """
    grid_size = int(math.sqrt(n))
    step_x = width / grid_size
    step_y = height / grid_size

    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            base_x = i * step_x + step_x / 2
            base_y = j * step_y + step_y / 2

            jitter_x = (np.random.random() - 0.5) * step_x * jitter
            jitter_y = (np.random.random() - 0.5) * step_y * jitter

            x = int(base_x + jitter_x)
            y = int(base_y + jitter_y)

            if 0 <= x < width and 0 <= y < height:
                positions.append((x, y))

    # Fill remaining if needed
    while len(positions) < n:
        positions.append((np.random.randint(0, width), np.random.randint(0, height)))

    return positions[:n]


# Pattern registry
PATTERN_GENERATORS = {
    'gaussian': generate_gaussian,
    'uniform': generate_uniform,
    'poisson': generate_poisson_disk,
    'spiral': generate_spiral,
    'concentric': generate_concentric,
    'halton': generate_halton,
    'jittered_grid': generate_jittered_grid,
    'grid': generate_grid,
}
