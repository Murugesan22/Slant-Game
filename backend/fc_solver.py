"""
Forward Checking (FC) Backtracking Solver for Slant Game

Uses constraint propagation after each assignment to proactively prune the
domains of unassigned neighbouring cells.  If any cell's domain becomes empty
the solver backtracks immediately — without ever recursing into a dead-end.

Algorithm Paradigm: Backtracking with forward checking (look-ahead)
- Cells are ordered in row-major order
- Each unassigned cell maintains a domain: a subset of {'L', 'R'}
- After placing a move, the solver checks every unassigned cell that shares
  a node with the placed cell and removes any move from its domain that would
  violate degree constraints or create a cycle
- If any domain becomes empty, the assignment is undone immediately (backtrack)
- Domains are restored on backtrack via snapshot copies

Time Complexity: O(2^(n*n)) worst case, but forward checking prunes much faster
Space Complexity: O(n*n) for domain snapshots and recursion stack
"""

import copy


# ==============================================================================
# Helper: which cells share a node with cell (r, c)?
# ==============================================================================

def _neighbours(r, c, size):
    """
    Return the list of cell coordinates that share at least one corner node
    with cell (r, c).

    A cell (r, c) touches 4 nodes: (r,c), (r,c+1), (r+1,c), (r+1,c+1).
    Any other cell that also touches one of these nodes is a neighbour.
    """
    nbrs = set()
    # Each node is shared by up to 4 cells.  For every node of (r, c),
    # the cells touching that node are offset by (-1,-1), (-1,0), (0,-1), (0,0)
    for nr, nc in [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]:
        for dr, dc in [(-1, -1), (-1, 0), (0, -1), (0, 0)]:
            cr, cc = nr + dr, nc + dc
            if 0 <= cr < size and 0 <= cc < size and (cr, cc) != (r, c):
                nbrs.add((cr, cc))
    return list(nbrs)


# ==============================================================================
# Core solver
# ==============================================================================

def solve_with_fc(game):
    """
    Solve the Slant game using Forward Checking backtracking.

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

    # Build initial domains — every cell starts with {'L', 'R'}
    domains = {}
    cells = []
    for r in range(size):
        for c in range(size):
            cells.append((r, c))
            domains[(r, c)] = ['L', 'R']

    success = _fc_backtrack(game, cells, 0, domains)

    if success:
        game.detect_cycle_dfs()
        game.check_completion()

    return success


def _fc_backtrack(game, cells, idx, domains):
    """
    Recursive forward-checking backtrack.

    Returns:
        bool: True if a valid solution was found
    """
    # Base case: all cells assigned
    if idx >= len(cells):
        return _verify_solution(game)

    r, c = cells[idx]

    # Try each value still in this cell's domain
    for mv in list(domains[(r, c)]):
        if not game.is_move_valid(r, c, mv):
            continue

        # Snapshot neighbour domains before propagation
        snapshot = {}
        neighbours = _neighbours(r, c, game.size)
        for nr, nc in neighbours:
            if game.grid[nr][nc] is None:
                snapshot[(nr, nc)] = list(domains[(nr, nc)])

        # Place the move
        game.apply_move(r, c, mv, check_validity=False, player='HUMAN')

        # Forward check: prune neighbour domains
        wipeout = _forward_check(game, cells, idx, domains, r, c, neighbours)

        if not wipeout:
            # No domain emptied → recurse
            if _fc_backtrack(game, cells, idx + 1, domains):
                return True

        # Undo
        game.undo()

        # Restore neighbour domains from snapshot
        for key, val in snapshot.items():
            domains[key] = val

    return False


def _forward_check(game, cells, idx, domains, r, c, neighbours):
    """
    After placing a move at (r, c), prune the domains of neighbouring
    unassigned cells.

    Returns:
        bool: True if a wipeout occurred (some domain became empty)
    """
    for nr, nc in neighbours:
        if game.grid[nr][nc] is not None:
            continue  # already assigned

        new_domain = []
        for mv in domains[(nr, nc)]:
            if game.is_move_valid(nr, nc, mv):
                new_domain.append(mv)
        domains[(nr, nc)] = new_domain

        if len(new_domain) == 0:
            return True  # wipeout — dead end

    return False


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
    Solve using Forward Checking and extract the solution grid.

    Args:
        game: SlantGame instance

    Returns:
        tuple: (success: bool, solution_grid: list or None)
    """
    game_copy = copy.deepcopy(game)

    success = solve_with_fc(game_copy)

    if success:
        return True, [row[:] for row in game_copy.grid]
    else:
        return False, None


# ==============================================================================
# RECORDED VARIANT (for visualization)
# ==============================================================================

def _fc_backtrack_recorded(game, cells, idx, domains, steps):
    """
    Identical to _fc_backtrack but records each decision point as a step dict.
    """
    # Base case
    if idx >= len(cells):
        if _verify_solution(game):
            steps.append({'action': 'SOLVED'})
            return True
        return False

    r, c = cells[idx]

    for mv in list(domains[(r, c)]):
        steps.append({'action': 'TRY', 'r': r, 'c': c, 'mv': mv})

        if not game.is_move_valid(r, c, mv):
            steps.append({'action': 'INVALID', 'r': r, 'c': c, 'mv': mv})
            continue

        # Snapshot
        snapshot = {}
        neighbours = _neighbours(r, c, game.size)
        for nr, nc in neighbours:
            if game.grid[nr][nc] is None:
                snapshot[(nr, nc)] = list(domains[(nr, nc)])

        # Place
        game.apply_move(r, c, mv, check_validity=False, player='HUMAN')

        # Forward check
        wipeout = False
        for nr, nc in neighbours:
            if game.grid[nr][nc] is not None:
                continue
            new_domain = []
            for nmv in domains[(nr, nc)]:
                if game.is_move_valid(nr, nc, nmv):
                    new_domain.append(nmv)

            pruned = set(domains[(nr, nc)]) - set(new_domain)
            domains[(nr, nc)] = new_domain

            if pruned:
                steps.append({
                    'action': 'PRUNE',
                    'r': nr, 'c': nc,
                    'removed': list(pruned),
                    'remaining': list(new_domain),
                    'cause_r': r, 'cause_c': c, 'cause_mv': mv
                })

            if len(new_domain) == 0:
                steps.append({'action': 'DEAD_END', 'r': nr, 'c': nc})
                wipeout = True
                break

        if not wipeout:
            if _fc_backtrack_recorded(game, cells, idx + 1, domains, steps):
                steps.append({'action': 'PLACE', 'r': r, 'c': c, 'mv': mv})
                return True

        steps.append({'action': 'BACKTRACK', 'r': r, 'c': c, 'mv': mv})
        game.undo()

        # Restore
        for key, val in snapshot.items():
            domains[key] = val

    return False


def solve_with_fc_recorded(game):
    """
    Solve using Forward Checking and record every step for visualization.

    Args:
        game: SlantGame instance

    Returns:
        tuple: (success: bool, steps: list of step dicts)
    """
    steps = []
    game_copy = copy.deepcopy(game)

    # Clear the grid
    size = game_copy.size
    for r in range(size):
        for c in range(size):
            if game_copy.grid[r][c] is not None:
                game_copy.remove_move(r, c)
    game_copy.history = []
    game_copy.scores = {'HUMAN': 0, 'CPU': 0}
    game_copy.owners = [[None for _ in range(size)] for _ in range(size)]

    # Build initial domains
    domains = {}
    cells = []
    for r in range(size):
        for c in range(size):
            cells.append((r, c))
            domains[(r, c)] = ['L', 'R']

    if not cells:
        return _verify_solution(game_copy), steps

    success = _fc_backtrack_recorded(game_copy, cells, 0, domains, steps)
    return success, steps


def get_fc_stats(steps):
    """Compute summary statistics from recorded FC steps."""
    return {
        'total_steps': len(steps),
        'backtracks': sum(1 for s in steps if s['action'] == 'BACKTRACK'),
        'prunes': sum(1 for s in steps if s['action'] == 'PRUNE'),
        'dead_ends': sum(1 for s in steps if s['action'] == 'DEAD_END'),
        'placements': sum(1 for s in steps if s['action'] == 'PLACE'),
    }
