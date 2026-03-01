"""
Conflict-Directed Backjumping (CBJ) Solver for Slant Game

Instead of chronological backtracking (undo one step at a time), CBJ identifies
which earlier decision caused the current failure and jumps directly back to it.
This avoids wasting time on cells that are irrelevant to the conflict.

Algorithm Paradigm: Backjumping with conflict-directed learning
- Cells are ordered in row-major order
- Each cell maintains a conflict set: indices of earlier cells that constrain it
- On failure, jump back to the most recent conflicting cell instead of the previous cell
- Conflict sets are merged when jumping past intermediate cells

Time Complexity: O(2^(n*n)) worst case, but backjumping prunes much faster than chronological
Space Complexity: O(n*n) for conflict sets and recursion stack
"""

import copy


def solve_with_cbj(game):
    """
    Solve the Slant game using Conflict-Directed Backjumping.

    Args:
        game: SlantGame instance (will be modified in-place if solution found)

    Returns:
        bool: True if solved, False if no solution exists
    """
    size = game.size

    # Clear the grid first (in case partially filled)
    for r in range(size):
        for c in range(size):
            if game.grid[r][c] is not None:
                game.remove_move(r, c)

    # Reset state
    game.history = []
    game.scores = {'HUMAN': 0, 'CPU': 0}
    game.owners = [[None for _ in range(size)] for _ in range(size)]

    # Collect all empty cells in row-major order
    cells = []
    for r in range(size):
        for c in range(size):
            cells.append((r, c))

    # Initialize conflict sets: one empty set per cell
    conflict_sets = [set() for _ in range(len(cells))]

    success, _ = _cbj_search(game, cells, 0, conflict_sets)

    if success:
        game.detect_cycle_dfs()
        game.check_completion()

    return success


def _cbj_search(game, cells, idx, conflict_sets):
    """
    Recursive CBJ search.

    Returns:
        tuple: (success: bool, jump_target: int)
    """
    # Base case: all cells processed
    if idx >= len(cells):
        if _verify_solution(game):
            return (True, -1)
        else:
            return (False, idx - 1)

    r, c = cells[idx]

    # Reset conflict set for this cell
    conflict_sets[idx] = set()

    for mv in ['L', 'R']:
        if not game.is_move_valid(r, c, mv):
            # Move is invalid — record the true conflict source
            conflict_source = _find_conflict_source(game, cells, r, c, mv, idx)
            if conflict_source >= 0:
                conflict_sets[idx].add(conflict_source)
            continue

        # Valid move — apply and recurse
        game.apply_move(r, c, mv, check_validity=False, player='HUMAN')

        success, jump_target = _cbj_search(game, cells, idx + 1, conflict_sets)

        if success:
            return (True, -1)

        # Merge conflict info from the child into our own conflict set
        # (exclude self-references so we don't blame ourselves)
        if idx + 1 < len(conflict_sets):
            conflict_sets[idx] |= (conflict_sets[idx + 1] - {idx})

        game.undo()

        # Classic CBJ optimisation: if the child's conflict doesn't involve
        # this cell at all (jump_target < idx), we can propagate immediately.
        # However this optimisation is only safe when conflict sources are
        # perfectly accurate.  Because our BFS identification can still miss
        # indirect causes, we stay conservative: only propagate when the
        # jump skips past ALL earlier cells (i.e. jump_target < 0), which
        # indicates an unrecoverable failure regardless of our choice here.
        if jump_target < 0:
            return (False, jump_target)

        # Otherwise keep trying remaining moves at this cell.
        # The child's conflict set has already been merged above, so if we
        # exhaust all values we will still jump to the right ancestor.

    # All moves exhausted for this cell
    if conflict_sets[idx]:
        jump_target = max(conflict_sets[idx])
    else:
        jump_target = idx - 1

    return (False, jump_target)


def _find_conflict_source(game, cells, r, c, mv, current_idx):
    """
    Find the most recent earlier cell that is a TRUE conflict source for
    placing move `mv` at (r, c).

    Two cases handled:
    1. Degree constraint violation: one of the move's endpoint nodes is already
       at its maximum degree. The conflict source is the most recent earlier cell
       that touches that SPECIFIC over-constrained node.
    2. Cycle violation: placing the diagonal would close a loop. The actual
       conflict sources are the cells forming the existing path between the two
       endpoint nodes. We use BFS to find that path and return the most recent
       cell index on it.

    Returns:
        int: Index of the most recent valid conflicting cell, or -1 if none found.
    """
    from collections import deque

    if mv == 'L':
        endpoint_a, endpoint_b = (r, c), (r + 1, c + 1)
    else:
        endpoint_a, endpoint_b = (r + 1, c), (r, c + 1)

    move_endpoints = {endpoint_a, endpoint_b}

    # ── Pass 1: Degree constraint violation ────────────────────────────────
    # If either endpoint node is already at its degree limit, the conflict is
    # any earlier cell that contributes an edge to that specific node.
    over_constrained_nodes = set()
    for node in move_endpoints:
        if node in game.constraints:
            if game.node_degrees.get(node, 0) >= game.constraints[node]:
                over_constrained_nodes.add(node)

    if over_constrained_nodes:
        # Find the most recent earlier cell touching an over-constrained node
        for i in range(current_idx - 1, -1, -1):
            er, ec = cells[i]
            if game.grid[er][ec] is None:
                continue
            placed_mv = game.grid[er][ec]
            if placed_mv == 'L':
                earlier_nodes = {(er, ec), (er + 1, ec + 1)}
            else:
                earlier_nodes = {(er + 1, ec), (er, ec + 1)}
            if earlier_nodes & over_constrained_nodes:
                return i
        return -1

    # ── Pass 2: Cycle violation ────────────────────────────────────────────
    # No degree violation → the move would create a cycle.
    # Find the existing path from endpoint_a to endpoint_b among placed cells.
    # The cells on that path are the true conflict sources.
    # Build adjacency: node -> list of (neighbour_node, cell_idx)
    adj = {}
    for i in range(current_idx):
        er, ec = cells[i]
        if game.grid[er][ec] is None:
            continue
        pmv = game.grid[er][ec]
        if pmv == 'L':
            n1, n2 = (er, ec), (er + 1, ec + 1)
        else:
            n1, n2 = (er + 1, ec), (er, ec + 1)
        adj.setdefault(n1, []).append((n2, i))
        adj.setdefault(n2, []).append((n1, i))

    # BFS: track (current_node, max_cell_index_on_path_so_far, visited)
    # We want the MOST RECENT (highest index) cell on the path, so we track
    # the running max as we traverse.
    queue = deque([(endpoint_a, -1, frozenset([endpoint_a]))])
    while queue:
        node, max_ci, visited = queue.popleft()
        for nxt, ci in adj.get(node, []):
            new_max = max(max_ci, ci)
            if nxt == endpoint_b:
                return new_max          # Most recent cell on the cycle path
            if nxt not in visited:
                queue.append((nxt, new_max, visited | {nxt}))

    # No path found (no cycle detected) — shouldn't reach here
    return -1


def _verify_solution(game):
    """
    Verify that the current game state is a valid, complete solution.

    Checks:
    1. All cells are filled
    2. All constraints are satisfied (node_degrees match constraints)
    3. No cycles exist

    Returns:
        bool: True if valid solution
    """
    size = game.size

    # 1. All cells must be filled
    for r in range(size):
        for c in range(size):
            if game.grid[r][c] is None:
                return False

    # 2. All constraints must match node degrees
    for node, limit in game.constraints.items():
        if game.node_degrees.get(node, 0) != limit:
            return False

    # 3. No cycles
    if game.detect_cycle_dfs():
        return False

    return True


def solve_and_extract(game):
    """
    Solve using CBJ and extract the solution grid.

    Args:
        game: SlantGame instance

    Returns:
        tuple: (success: bool, solution_grid: list or None)
    """
    # Create a copy to avoid modifying original
    game_copy = copy.deepcopy(game)

    success = solve_with_cbj(game_copy)

    if success:
        return True, [row[:] for row in game_copy.grid]
    else:
        return False, None

