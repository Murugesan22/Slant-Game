"""
Hybrid Solver for Slant Game

Combines Divide & Conquer partitioning with Dynamic Programming solving.

Algorithm:
1. DIVIDE: Use D&C to partition grid into subgrids (quadrants)
2. CONQUER: Use DP to solve each subgrid independently
3. COMBINE: Merge subgrid solutions respecting global constraints

NO greedy heuristics allowed - strictly algorithmic approach.
"""

from dp_solver import solve_with_dp_enhanced, MemoizedValidator
import copy


def solve_with_hybrid(game):
    """
    Solve the Slant game using Hybrid D&C + DP approach.
    
    Strategy:
    1. Divide the grid into quadrants using D&C partitioning
    2. Solve each quadrant using DP (with memoization)
    3. Merge solutions ensuring global constraint satisfaction
    4. Backtrack if merging fails, try different quadrant solutions
    
    Args:
        game: SlantGame instance (modified in-place if solution found)
    
    Returns:
        bool: True if solved, False if no solution exists
    """
    size = game.size
    
    # Base case: small grids use pure DP
    if size <= 3:
        return solve_with_dp_enhanced(game)
    
    # Divide into quadrants
    mid_r = size // 2
    mid_c = size // 2
    
    quadrants = [
        ('TL', 0, mid_r, 0, mid_c),           # Top-left
        ('TR', 0, mid_r, mid_c, size),        # Top-right
        ('BL', mid_r, size, 0, mid_c),        # Bottom-left
        ('BR', mid_r, size, mid_c, size),     # Bottom-right
    ]
    
    # Try to solve quadrants sequentially with backtracking
    return _solve_quadrants_hybrid(game, quadrants, 0)


def _solve_quadrants_hybrid(game, quadrants, q_idx):
    """
    Recursively solve quadrants using DP, with backtracking on failure.
    
    This is the COMBINE step: each quadrant's solution must be
    compatible with all subsequent quadrants.
    
    Args:
        game: SlantGame instance
        quadrants: List of (name, r_start, r_end, c_start, c_end)
        q_idx: Current quadrant index
    
    Returns:
        bool: True if all quadrants solved successfully
    """
    if q_idx >= len(quadrants):
        # All quadrants filled - verify global solution
        return _verify_global_solution(game)
    
    name, r_start, r_end, c_start, c_end = quadrants[q_idx]
    cells = []
    
    # Collect cells in this quadrant
    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            if game.grid[r][c] is None:
                cells.append((r, c))
    
    # If quadrant already filled, move to next
    if not cells:
        return _solve_quadrants_hybrid(game, quadrants, q_idx + 1)
    
    # Use DP to solve this quadrant with backtracking
    return _dp_solve_quadrant_with_continuation(game, cells, 0, quadrants, q_idx)


def _dp_solve_quadrant_with_continuation(game, cells, cell_idx, quadrants, q_idx):
    """
    DP-based backtracking for current quadrant with continuation to next quadrants.
    
    Uses memoized validity checks for efficiency (DP principle).
    After filling current quadrant, attempts to solve remaining quadrants.
    
    Args:
        game: SlantGame instance
        cells: List of (r, c) in current quadrant
        cell_idx: Current cell index
        quadrants: All quadrants
        q_idx: Current quadrant index
    
    Returns:
        bool: True if this quadrant and all subsequent quadrants solved
    """
    if cell_idx >= len(cells):
        # Current quadrant filled - try next quadrant
        return _solve_quadrants_hybrid(game, quadrants, q_idx + 1)
    
    r, c = cells[cell_idx]
    
    # Try both moves with DP memoization
    validator = MemoizedValidator(game)
    
    for move_type in ['L', 'R']:
        if validator.is_move_valid_cached(r, c, move_type):
            game.apply_move(r, c, move_type, check_validity=False)
            
            # Check if this move allows continuation
            if _dp_solve_quadrant_with_continuation(game, cells, cell_idx + 1, quadrants, q_idx):
                return True
            
            game.undo()
            validator.invalidate(r, c)
    
    return False


def _verify_global_solution(game):
    """
    Verify the complete grid is a valid solution.
    
    Checks:
    1. All cells filled
    2. All constraints satisfied
    3. No cycles exist
    
    Returns:
        bool: True if valid solution
    """
    # Check all cells filled
    for r in range(game.size):
        for c in range(game.size):
            if game.grid[r][c] is None:
                return False
    
    # Check all constraints satisfied
    for node, limit in game.constraints.items():
        if game.node_degrees[node] != limit:
            return False
    
    # Check no cycles
    if game.detect_cycle_dfs():
        return False
    
    return True


def extract_solution_grid(game):
    """
    Extract the solution grid from a solved game.
    
    Args:
        game: Solved SlantGame instance
    
    Returns:
        list: 2D grid of solution moves ('L' or 'R')
    """
    return [row[:] for row in game.grid]


def solve_and_extract(game):
    """
    Solve the game using hybrid approach and extract solution grid.
    
    Args:
        game: SlantGame instance
    
    Returns:
        tuple: (success: bool, solution_grid: list or None)
    """
    # Create a copy to avoid modifying original
    game_copy = copy.deepcopy(game)
    
    success = solve_with_hybrid(game_copy)
    
    if success:
        return True, extract_solution_grid(game_copy)
    else:
        return False, None
