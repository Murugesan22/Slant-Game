"""
Cut-based Partition Solver for Slant Game

Identifies independent constraint regions and solves recursively.

Algorithm:
1. IDENTIFY: Find independent constraint regions (minimal cuts)
2. PARTITION: Recursively partition based on constraint boundaries
3. SOLVE: Solve each region independently
4. COMBINE: Merge regional solutions

NO local scoring or greedy evaluation - strictly algorithmic approach.
"""

import copy
from collections import deque


def solve_with_cut(game):
    """
    Solve the Slant game using Cut-based Partition approach.
    
    Strategy:
    1. Identify constraint regions (areas with dense constraints)
    2. Find minimal cuts that separate regions
    3. Recursively solve each region
    4. Combine solutions respecting boundary constraints
    
    Args:
        game: SlantGame instance (modified in-place if solution found)
    
    Returns:
        bool: True if solved, False if no solution exists
    """
    size = game.size
    
    # Base case: small grids use direct backtracking
    if size <= 3:
        return _backtrack_solve(game)
    
    # Find optimal cut line
    cut_info = _find_optimal_cut(game)
    
    if cut_info is None:
        # No good cut found - use direct backtracking
        return _backtrack_solve(game)
    
    cut_type, cut_pos = cut_info
    
    # Partition based on cut
    regions = _partition_by_cut(game, cut_type, cut_pos)
    
    # Solve regions recursively
    return _solve_regions_sequential(game, regions, 0)


def _find_optimal_cut(game):
    """
    Find the optimal cut line that minimizes boundary constraints.
    
    Scans all possible horizontal and vertical cuts, scoring each by:
    - Number of constraints on the cut line (lower is better)
    - Balance of region sizes (more balanced is better)
    
    Returns:
        tuple: (cut_type, cut_position) or None if no good cut
               cut_type: 'H' (horizontal) or 'V' (vertical)
               cut_position: row/col index of cut
    """
    size = game.size
    best_cut = None
    best_score = float('inf')
    
    # Try horizontal cuts
    for r in range(1, size):
        score = _score_horizontal_cut(game, r)
        if score < best_score:
            best_score = score
            best_cut = ('H', r)
    
    # Try vertical cuts
    for c in range(1, size):
        score = _score_vertical_cut(game, c)
        if score < best_score:
            best_score = score
            best_cut = ('V', c)
    
    # Only return cut if it's actually beneficial
    # (i.e., significantly reduces problem complexity)
    if best_score < size * 2:  # Heuristic threshold
        return best_cut
    
    return None


def _score_horizontal_cut(game, row):
    """
    Score a horizontal cut at the given row.
    
    Lower score = better cut (fewer boundary constraints, more balanced regions)
    
    Args:
        game: SlantGame instance
        row: Row index for horizontal cut
    
    Returns:
        float: Cut score
    """
    size = game.size
    
    # Count constraints on boundary nodes
    boundary_constraints = 0
    for c in range(size + 1):
        node = (row, c)
        if node in game.constraints:
            boundary_constraints += 1
    
    # Calculate region size imbalance
    top_size = row
    bottom_size = size - row
    imbalance = abs(top_size - bottom_size)
    
    # Combined score (lower is better)
    return boundary_constraints * 2 + imbalance


def _score_vertical_cut(game, col):
    """
    Score a vertical cut at the given column.
    
    Lower score = better cut (fewer boundary constraints, more balanced regions)
    
    Args:
        game: SlantGame instance
        col: Column index for vertical cut
    
    Returns:
        float: Cut score
    """
    size = game.size
    
    # Count constraints on boundary nodes
    boundary_constraints = 0
    for r in range(size + 1):
        node = (r, col)
        if node in game.constraints:
            boundary_constraints += 1
    
    # Calculate region size imbalance
    left_size = col
    right_size = size - col
    imbalance = abs(left_size - right_size)
    
    # Combined score (lower is better)
    return boundary_constraints * 2 + imbalance


def _partition_by_cut(game, cut_type, cut_pos):
    """
    Partition the grid based on cut line.
    
    Args:
        game: SlantGame instance
        cut_type: 'H' (horizontal) or 'V' (vertical)
        cut_pos: Row/column index of cut
    
    Returns:
        list: List of regions, each region is a list of (r, c) cells
    """
    size = game.size
    regions = []
    
    if cut_type == 'H':
        # Horizontal cut - split into top and bottom
        top_cells = []
        bottom_cells = []
        
        for r in range(size):
            for c in range(size):
                if game.grid[r][c] is None:
                    if r < cut_pos:
                        top_cells.append((r, c))
                    else:
                        bottom_cells.append((r, c))
        
        if top_cells:
            regions.append(top_cells)
        if bottom_cells:
            regions.append(bottom_cells)
    
    else:  # cut_type == 'V'
        # Vertical cut - split into left and right
        left_cells = []
        right_cells = []
        
        for r in range(size):
            for c in range(size):
                if game.grid[r][c] is None:
                    if c < cut_pos:
                        left_cells.append((r, c))
                    else:
                        right_cells.append((r, c))
        
        if left_cells:
            regions.append(left_cells)
        if right_cells:
            regions.append(right_cells)
    
    return regions


def _solve_regions_sequential(game, regions, region_idx):
    """
    Solve regions one at a time with backtracking.
    
    If a later region fails, backtrack the current region and try different fill.
    
    Args:
        game: SlantGame instance
        regions: List of cell regions
        region_idx: Current region index
    
    Returns:
        bool: True if all regions solved successfully
    """
    if region_idx >= len(regions):
        # All regions solved - verify global solution
        return _verify_solution(game)
    
    region_cells = regions[region_idx]
    
    # Solve current region with continuation to next regions
    return _backtrack_region_with_continuation(game, region_cells, 0, regions, region_idx)


def _backtrack_region_with_continuation(game, cells, cell_idx, regions, region_idx):
    """
    Backtracking fill for current region with continuation to next regions.
    
    Args:
        game: SlantGame instance
        cells: Cells in current region
        cell_idx: Current cell index
        regions: All regions
        region_idx: Current region index
    
    Returns:
        bool: True if this region and all subsequent regions solved
    """
    if cell_idx >= len(cells):
        # Current region filled - try next region
        return _solve_regions_sequential(game, regions, region_idx + 1)
    
    r, c = cells[cell_idx]
    
    for move_type in ['L', 'R']:
        if game.is_move_valid(r, c, move_type):
            game.apply_move(r, c, move_type, check_validity=False)
            
            if _backtrack_region_with_continuation(game, cells, cell_idx + 1, regions, region_idx):
                return True
            
            game.undo()
    
    return False


def _backtrack_solve(game):
    """
    Simple backtracking solver for small grids or when no good cut exists.
    
    Args:
        game: SlantGame instance
    
    Returns:
        bool: True if solved
    """
    # Find first empty cell
    empty_cell = None
    for r in range(game.size):
        for c in range(game.size):
            if game.grid[r][c] is None:
                empty_cell = (r, c)
                break
        if empty_cell:
            break
    
    if empty_cell is None:
        # All filled - verify solution
        return _verify_solution(game)
    
    r, c = empty_cell
    
    for move_type in ['L', 'R']:
        if game.is_move_valid(r, c, move_type):
            game.apply_move(r, c, move_type, check_validity=False)
            
            if _backtrack_solve(game):
                return True
            
            game.undo()
    
    return False


def _verify_solution(game):
    """
    Verify the complete grid is a valid solution.
    
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
    Solve the game using cut-based approach and extract solution grid.
    
    Args:
        game: SlantGame instance
    
    Returns:
        tuple: (success: bool, solution_grid: list or None)
    """
    # Create a copy to avoid modifying original
    game_copy = copy.deepcopy(game)
    
    success = solve_with_cut(game_copy)
    
    if success:
        return True, extract_solution_grid(game_copy)
    else:
        return False, None
