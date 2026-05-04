# Jacob Rodas, CPSC 481, Logistics Robot Decision Agent - Search Module
"""
search.py - Search Engineer component.

Responsibilities:
    - Define the warehouse grid (S, G, #, H, R, .)
    - Generate candidate routes from S to G using BFS
    - Return route metadata (path, steps, zones touched) for the ML

Strategy for generating multiple candidate routes:
    1. Run BFS with several different MOVE_ORDERs. Different exploration
       orders cause BFS to break ties differently when multiple equal-
       length shortest paths exist, producing distinct routes.
    2. If fewer than the requested number of distinct routes are found,
       fall back to BFS with node exclusion: pick a mid-cell from an
       existing route, mark it as blocked, and re-run BFS to force a
       different path.

    All routes returned are still found by BFS.

Symbol legend:
    S = Start
    G = Goal
    # = Wall (impassable)
    R = Restricted zone (traversable, flagged)
    H = High-traffic zone (traversable, flagged)
    . = Normal open cell
"""

from collections import deque

# Warehouse grid. R and H are traversable; only # blocks movement.
# Search reports which zones each route touches; the reasoning module
# decides whether to reject or penalize based on that information.
GRID = [
    ['S', '.', '.', '.', '.', '.'],
    ['.', '#', '#', 'H', 'H', '.'],
    ['.', '.', '.', '.', '.', '.'],
    ['.', '.', '#', '#', '.', '.'],
    ['R', 'R', '.', '.', '.', '#'],
    ['.', '.', '.', '#', '.', 'G'],
]

# Direction map
DIRECTION_MAP = {
    "UP":    (-1, 0),
    "DOWN":  (1, 0),
    "LEFT":  (0, -1),
    "RIGHT": (0, 1),
}

# Default move order for the primary BFS run. Additional orders are used
# to encourage diverse candidate paths.
DEFAULT_MOVE_ORDER = ["UP", "RIGHT", "DOWN", "LEFT"]

# Move orders tried (in order) when generating multiple candidates.
CANDIDATE_MOVE_ORDERS = [
    ["UP", "RIGHT", "DOWN", "LEFT"],
    ["DOWN", "RIGHT", "UP", "LEFT"],
    ["RIGHT", "DOWN", "LEFT", "UP"],
    ["LEFT", "DOWN", "RIGHT", "UP"],
]


# -------------------------------------------------------------------
# Grid helpers
# -------------------------------------------------------------------

def find_position(grid, symbol):
    """Return the (row, col) of the first cell matching symbol, or None."""
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == symbol:
                return (r, c)
    return None


def get_neighbors(grid, pos, move_order, blocked=None):
    """
    Return legal neighbor positions of pos, in the given move_order.

    A neighbor is legal if it is inside the grid, is not a wall (#),
    and is not in the optional `blocked` set (used for the exclusion
    fallback when generating diverse candidate routes).
    """
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


# -------------------------------------------------------------------
# BFS
# -------------------------------------------------------------------

def bfs(grid, move_order=None, blocked=None):
    """
    Standard BFS from S to G.

    Returns:
        (path, nodes_expanded)
        path is a list of (row, col) tuples from S to G, or None if no
        path exists. nodes_expanded is the number of states dequeued.

    Optional `blocked` is a set of (row, col) cells to treat as walls
    for this run only (used by the exclusion fallback).
    """
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
            # Reconstruct path by walking parent pointers back to S.
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


# -------------------------------------------------------------------
# Candidate route generation
# -------------------------------------------------------------------

def describe_route(grid, path, route_id):
    """
    Build a route descriptor for the ML / reasoning modules.

    Returned dict contains everything downstream needs without forcing
    them to re-walk the path:

        id                  - human-readable name like "Route A"
        path                - list of (row, col) tuples from S to G
        steps               - number of moves (len(path) - 1)
        zones_touched       - list of zone symbols encountered in order
                              (excluding S, G, .) - duplicates kept so
                              that callers can use raw counts to
                              estimate congestion
        passes_restricted   - True if the path crosses any R cell
        passes_high_traffic - True if the path crosses any H cell
        high_traffic_count  - number of H cells crossed
    """
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
    }


def find_candidate_paths(grid, max_paths=3):
    """
    Generate up to max_paths distinct candidate routes from S to G.

    Hybrid strategy:
        1. Try several MOVE_ORDERs; different orderings make BFS break
           ties differently when multiple shortest paths exist.
        2. If we still have fewer than max_paths distinct routes,
           fall back to BFS with node exclusion: temporarily block a
           mid-cell from an existing path and re-run BFS.

    Returns a list of route descriptors (see describe_route).
    """
    start = find_position(grid, 'S')
    goal = find_position(grid, 'G')

    distinct_paths = []

    # Phase 1: vary MOVE_ORDER
    for order in CANDIDATE_MOVE_ORDERS:
        if len(distinct_paths) >= max_paths:
            break
        path, _ = bfs(grid, move_order=order)
        if path is not None and path not in distinct_paths:
            distinct_paths.append(path)

    # Phase 2: node-exclusion fallback if we still need more routes
    if len(distinct_paths) < max_paths:
        for existing in list(distinct_paths):
            if len(distinct_paths) >= max_paths:
                break
            # Try blocking each interior cell of an existing path.
            interior = [c for c in existing if c != start and c != goal]
            for cell in interior:
                if len(distinct_paths) >= max_paths:
                    break
                path, _ = bfs(grid, move_order=DEFAULT_MOVE_ORDER,
                              blocked={cell})
                if path is not None and path not in distinct_paths:
                    distinct_paths.append(path)

    # Wrap each path in a descriptor for downstream modules.
    routes = []
    for i, path in enumerate(distinct_paths):
        route_id = f"Route {chr(ord('A') + i)}"
        routes.append(describe_route(grid, path, route_id))
    return routes


# -------------------------------------------------------------------
# Standalone demo - runs when search.py is executed directly.
# main.py imports find_candidate_paths and GRID and drives the full
# pipeline; this block is only for testing the search component on
# its own.
# -------------------------------------------------------------------

def _zone_label(route):
    """Short human-readable zone summary for one route."""
    if route["passes_restricted"]:
        return "restricted"
    if route["passes_high_traffic"]:
        return "high_traffic"
    return "normal"


def main():
    print("Warehouse grid:")
    print_grid(GRID)

    routes = find_candidate_paths(GRID, max_paths=3)

    if not routes:
        print("No routes found from S to G.")
        return

    print(f"Generated {len(routes)} candidate route(s):\n")
    for route in routes:
        print(f"{route['id']}")
        print(f"  Path:        {route['path']}")
        print(f"  Steps:       {route['steps']}")
        print(f"  Zone type:   {_zone_label(route)}")
        print(f"  Touches H:   {route['high_traffic_count']}")
        print(f"  Touches R:   {route['passes_restricted']}")
        print()

        # Visualize this route on a grid copy.
        marked = [row[:] for row in GRID]
        for (r, c) in route['path']:
            if marked[r][c] not in ('S', 'G', 'H', 'R'):
                marked[r][c] = '*'
        print_grid(marked)


if __name__ == "__main__":
    main()