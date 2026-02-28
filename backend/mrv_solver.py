"""
MRV (Minimum Remaining Values) Backtracking Solver for Slant Game

Uses the MRV heuristic to always pick the most constrained empty cell first,
dramatically reducing the search space compared to naive left-to-right ordering.

Algorithm Paradigm: Constraint-satisfaction backtracking with MRV ordering
- At each step, select the empty cell with the fewest valid moves
- Order valid moves by how many constraints they satisfy (most satisfying first)
- Apply move, recurse, undo on failure

Time Complexity: O(2^(n*n)) worst case, but MRV pruning makes it much faster in practice
Space Complexity: O(n*n) for recursion stack
"""

import copy


def solve_with_mrv(game):
    """
    Solve the Slant game using MRV backtracking.

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

    result = _mrv_backtrack(game)

    if result:
        game.detect_cycle_dfs()
        game.check_completion()

    return result


def _get_mrv_cell(game):
    """
    Find the empty cell with the minimum number of remaining valid moves (MRV).

    Returns:
        tuple: (r, c, count) for the most constrained cell, or None if no empty cells
    """
    size = game.size
    best_cell = None
    best_count = float('inf')

    for r in range(size):
        for c in range(size):
            if game.grid[r][c] is not None:
                continue

            # Count valid moves for this cell
            count = 0
            for mv in ['L', 'R']:
                if game.is_move_valid(r, c, mv):
                    count += 1

            # Dead end: cell has no valid moves — return immediately
            if count == 0:
                return (r, c, 0)

            # Forced move: only one option — return immediately
            if count == 1:
                return (r, c, 1)

            if count < best_count:
                best_count = count
                best_cell = (r, c, count)

    return best_cell


def _get_valid_moves(game, r, c):
    """
    Get valid moves for cell (r, c), ordered by how many constraints they satisfy
    (most satisfying first).

    A move 'L' uses nodes (r, c) and (r+1, c+1).
    A move 'R' uses nodes (r+1, c) and (r, c+1).

    Args:
        game: SlantGame instance
        r, c: Cell coordinates

    Returns:
        list: Valid move types ordered by constraint satisfaction count (descending)
    """
    valid_moves = []

    for mv in ['L', 'R']:
        if not game.is_move_valid(r, c, mv):
            continue

        # Count how many constraints this move satisfies
        if mv == 'L':
            nodes = [(r, c), (r + 1, c + 1)]
        else:
            nodes = [(r + 1, c), (r, c + 1)]

        satisfaction = 0
        for node in nodes:
            if node in game.constraints:
                # After placing this move, degree increases by 1
                new_deg = game.node_degrees[node] + 1
                if new_deg == game.constraints[node]:
                    satisfaction += 1

        valid_moves.append((mv, satisfaction))

    # Sort by satisfaction count descending (most satisfying first)
    valid_moves.sort(key=lambda x: x[1], reverse=True)

    return [mv for mv, _ in valid_moves]


def _mrv_backtrack(game):
    """
    Core MRV backtracking recursion.

    At each step:
    1. Find the most constrained empty cell (MRV)
    2. Try valid moves in order of constraint satisfaction
    3. Apply, recurse, undo on failure

    Returns:
        bool: True if a valid solution was found
    """
    # Find the most constrained empty cell
    mrv_result = _get_mrv_cell(game)

    if mrv_result is None:
        # No empty cells — board is full, verify solution
        return _verify_solution(game)

    r, c, count = mrv_result

    # Dead end: no valid moves for this cell
    if count == 0:
        return False

    # Get valid moves ordered by constraint satisfaction
    valid_moves = _get_valid_moves(game, r, c)

    for mv in valid_moves:
        game.apply_move(r, c, mv, check_validity=False, player='HUMAN')

        if _mrv_backtrack(game):
            return True

        game.undo()

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
    Solve using MRV backtracking and extract the solution grid.

    Args:
        game: SlantGame instance

    Returns:
        tuple: (success: bool, solution_grid: list or None)
    """
    # Create a copy to avoid modifying original
    game_copy = copy.deepcopy(game)

    success = solve_with_mrv(game_copy)

    if success:
        return True, [row[:] for row in game_copy.grid]
    else:
        return False, None


# ==============================================================================
# RECORDED VARIANT (for visualization)
# ==============================================================================

def _mrv_backtrack_recorded(game, steps):
    """
    Identical to _mrv_backtrack but records each decision point as a step dict.
    """
    mrv_result = _get_mrv_cell(game)

    if mrv_result is None:
        # Board is full — verify
        if _verify_solution(game):
            steps.append({'action': 'SOLVED'})
            return True
        return False

    r, c, count = mrv_result

    # Record MRV selection
    steps.append({'action': 'MRV_SELECT', 'r': r, 'c': c, 'options': count})

    # Dead end
    if count == 0:
        steps.append({'action': 'DEAD_END', 'r': r, 'c': c})
        return False

    # Get valid moves
    valid_moves = _get_valid_moves(game, r, c)

    # Record forced move if only one option
    if count == 1 and valid_moves:
        steps.append({'action': 'FORCED', 'r': r, 'c': c, 'mv': valid_moves[0]})

    for mv in valid_moves:
        steps.append({'action': 'TRY', 'r': r, 'c': c, 'mv': mv})

        game.apply_move(r, c, mv, check_validity=False, player='HUMAN')

        if _mrv_backtrack_recorded(game, steps):
            steps.append({'action': 'PLACE', 'r': r, 'c': c, 'mv': mv})
            return True

        steps.append({'action': 'BACKTRACK', 'r': r, 'c': c, 'mv': mv})
        game.undo()

    return False


def solve_with_mrv_recorded(game):
    """
    Solve using MRV backtracking and record every step for visualization.

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

    success = _mrv_backtrack_recorded(game_copy, steps)
    return success, steps


def get_mrv_stats(steps):
    """Compute summary statistics from recorded MRV steps."""
    return {
        'total_steps': len(steps),
        'backtracks': sum(1 for s in steps if s['action'] == 'BACKTRACK'),
        'forced_moves': sum(1 for s in steps if s['action'] == 'FORCED'),
        'dead_ends': sum(1 for s in steps if s['action'] == 'DEAD_END'),
        'placements': sum(1 for s in steps if s['action'] == 'PLACE'),
    }

