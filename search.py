# Jacob Rodas, CPSC 481, Logistics Robot Decision Agent - Search Module
from collections import deque

GRID = [
    ['S', '.', '.', '.', '.', '.'],
    ['.', '#', '#', 'H', 'H', '.'],
    ['.', '.', '.', '.', '.', '.'],
    ['.', '.', '#', '#', '.', '.'],
    ['R', 'R', '.', '.', '.', '#'],
    ['.', '.', '.', '#', '.', 'G'],
]

DIRECTION_MAP = {
    "UP":    (-1, 0),
    "DOWN":  (1, 0),
    "LEFT":  (0, -1),
    "RIGHT": (0, 1),
}

DEFAULT_MOVE_ORDER = ["UP", "RIGHT", "DOWN", "LEFT"]

CANDIDATE_MOVE_ORDERS = [
    ["UP", "RIGHT", "DOWN", "LEFT"],
    ["DOWN", "RIGHT", "UP", "LEFT"],
    ["RIGHT", "DOWN", "LEFT", "UP"],
    ["LEFT", "DOWN", "RIGHT", "UP"],
]


def find_position(grid, symbol):
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == symbol:
                return (r, c)
    return None


def get_neighbors(grid, pos, move_order, blocked=None):
    blocked = blocked or set()
    r, c = pos
    rows, cols = len(grid), len(grid[0])
    neighbors = []
    for direction in move_order:
        dr, dc = DIRECTION_MAP[direction]
        nr, nc = r + dr, c + dc
        in_bounds = 0 <= nr < rows and 0 <= nc < cols
        if in_bounds and grid[nr][nc] != '#' and (nr, nc) not in blocked:
            neighbors.append((nr, nc))
    return neighbors


def print_grid(grid):
    for row in grid:
        print(' '.join(row))
    print()


def bfs(grid, move_order=None, blocked=None):
    if move_order is None:
        move_order = DEFAULT_MOVE_ORDER

    start = find_position(grid, 'S')
    goal = find_position(grid, 'G')
    if start is None or goal is None:
        return None, 0

    frontier = deque([start])
    visited = {start}
    parent = {start: None}
    nodes_expanded = 0

    while frontier:
        current = frontier.popleft()
        nodes_expanded += 1

        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, nodes_expanded

        for neighbor in get_neighbors(grid, current, move_order, blocked):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                frontier.append(neighbor)

    return None, nodes_expanded


def describe_route(grid, path, route_id):
    cells = [grid[r][c] for (r, c) in path]
    zones = [ch for ch in cells if ch in ('H', 'R')]
    return {
        "id": route_id,
        "path": path,
        "steps": len(path) - 1,
        "zones_touched": zones,
        "passes_restricted": 'R' in cells,
        "passes_high_traffic": 'H' in cells,
        "high_traffic_count": cells.count('H'),
        "restricted_count": cells.count('R'),
    }


def find_candidate_paths(grid, max_paths=3):
    start = find_position(grid, 'S')
    goal = find_position(grid, 'G')

    distinct_paths = []

    # Phase 1: vary MOVE_ORDER to break BFS ties differently
    for order in CANDIDATE_MOVE_ORDERS:
        if len(distinct_paths) >= max_paths:
            break
        path, _ = bfs(grid, move_order=order)
        if path is not None and path not in distinct_paths:
            distinct_paths.append(path)

    # Phase 2: node-exclusion fallback
    if len(distinct_paths) < max_paths:
        for existing in list(distinct_paths):
            if len(distinct_paths) >= max_paths:
                break
            interior = [c for c in existing if c != start and c != goal]
            for cell in interior:
                if len(distinct_paths) >= max_paths:
                    break
                path, _ = bfs(grid, move_order=DEFAULT_MOVE_ORDER,
                              blocked={cell})
                if path is not None and path not in distinct_paths:
                    distinct_paths.append(path)

    routes = []
    for i, path in enumerate(distinct_paths):
        route_id = f"Route {chr(ord('A') + i)}"
        routes.append(describe_route(grid, path, route_id))
    return routes