"""
Dynamic Programming Solver for Slant Game

Implements two DP-based approaches:
1. Profile DP Solver - Processes grid row-by-row with boundary state memoization
2. Memoized Constraint Validator - Caches validity checks to avoid redundant BFS

Algorithm Paradigm: Dynamic Programming
- Overlapping Subproblems: Same boundary profiles recur across different row assignments
- Optimal Substructure: A valid solution for rows 0..k extends to rows 0..k+1

Time Complexity: O(rows * 2^cols * |profiles|) where |profiles| is bounded by degree combinations
Space Complexity: O(2^cols * |profiles|) for memoization table
"""

import itertools


# ==============================================================================
# PROFILE DP SOLVER
# ==============================================================================

def solve_with_dp(game):
    """
    Solve the Slant game using Profile Dynamic Programming.
    
    Approach:
    - Process the grid row by row (top to bottom)
    - For each row, try all 2^size possible slash assignments (L/R per cell)
    - State = tuple of node degrees along the bottom boundary of the current row
    - Memoize: memo[(row, boundary_profile)] = the assignment that leads to a solution
    - Prune assignments that violate constraints or create cycles
    
    For boards up to 9x9, 2^9 = 512 combinations per row, which is very manageable.
    
    Args:
        game: SlantGame instance (will be modified in-place if solution found)
    
    Returns:
        bool: True if solved, False if no solution exists
    """
    size = game.size
    
    # Generate all possible row assignments: each cell is 'L' or 'R'
    # A row of size N has 2^N possible assignments
    all_row_assignments = list(itertools.product(['L', 'R'], repeat=size))
    
    # Memoization table: (row_index, boundary_state) -> assignment or None
    memo = {}
    
    # Track the solution path
    solution = [[None] * size for _ in range(size)]
    
    def get_boundary_state(row_idx):
        """
        Capture the boundary state after filling rows 0..row_idx.
        The boundary is the set of node degrees on row (row_idx + 1),
        which is the bottom edge of the current row and top edge of the next.
        
        Also includes constraint satisfaction status for fully determined nodes.
        """
        boundary_row = row_idx + 1
        state = []
        for c in range(size + 1):
            node = (boundary_row, c)
            deg = game.node_degrees[node]
            constraint = game.constraints.get(node, -1)
            state.append((deg, constraint))
        return tuple(state)
    
    def get_top_boundary_state():
        """Capture the degree state of the top row nodes (row 0)."""
        state = []
        for c in range(size + 1):
            node = (0, c)
            deg = game.node_degrees[node]
            constraint = game.constraints.get(node, -1)
            state.append((deg, constraint))
        return tuple(state)
    
    def check_row_constraints_valid(row_idx, assignment):
        """
        Check if placing this assignment in the given row violates any constraints.
        
        Checks:
        1. Top-row nodes (row_idx) - these are fully determined after this row
           if row_idx > 0 (they won't get more edges from below)
        2. Degree limits (max 4, and constraint limits)
        3. No cycles created
        """
        # Temporarily apply the assignment
        applied = []
        valid = True
        
        for c, move_type in enumerate(assignment):
            # Check if placing this move is valid
            if not game.is_move_valid(row_idx, c, move_type):
                valid = False
                break
            
            # Apply the move temporarily
            game.apply_move(row_idx, c, move_type, check_validity=False, player='HUMAN')
            applied.append(c)
        
        if not valid:
            # Undo all applied moves in this row
            for c in reversed(applied):
                game.undo()
            return False
        
        # Check top-row constraints (nodes at row_idx are now fully determined
        # because no future row will add edges to them — except the current row's
        # bottom boundary which IS row_idx+1)
        if row_idx > 0:
            for c in range(size + 1):
                node = (row_idx, c)
                if node in game.constraints:
                    # For interior nodes (not on top/bottom boundary of entire grid),
                    # they receive edges from the row above and the current row.
                    # After filling row_idx, nodes at position row_idx are fully determined
                    # (they've received all possible edges from rows row_idx-1 and row_idx).
                    deg = game.node_degrees[node]
                    limit = game.constraints[node]
                    if deg != limit:
                        # This node is fully determined but doesn't match constraint
                        valid = False
                        break
        
        if not valid:
            for c in reversed(applied):
                game.undo()
            return False
        
        return True
    
    def undo_row(row_idx, num_cells):
        """Undo all moves in a row."""
        for _ in range(num_cells):
            game.undo()
    
    def check_final_constraints():
        """After filling the entire grid, check bottom-row constraints."""
        for c in range(size + 1):
            node = (size, c)
            if node in game.constraints:
                if game.node_degrees[node] != game.constraints[node]:
                    return False
        # Also check first row constraints that weren't checked
        for c in range(size + 1):
            node = (0, c)
            if node in game.constraints:
                if game.node_degrees[node] != game.constraints[node]:
                    return False
        return True
    
    def dp_solve_row(row_idx):
        """
        DP recursive function: try to solve from row_idx to the last row.
        
        Uses memoization on (row_idx, boundary_state) to avoid recomputation.
        The boundary_state captures node degrees at the interface between
        row_idx-1 and row_idx, which determines what assignments are feasible.
        """
        if row_idx == size:
            # All rows filled — check final constraints
            return check_final_constraints()
        
        # Get current boundary state for memoization
        if row_idx == 0:
            boundary = get_top_boundary_state()
        else:
            boundary = get_boundary_state(row_idx - 1)
        
        memo_key = (row_idx, boundary)
        
        if memo_key in memo:
            cached = memo[memo_key]
            if cached is None:
                return False
            # Replay the cached solution
            for row in range(row_idx, size):
                for c, move_type in enumerate(cached[row]):
                    game.apply_move(row, c, move_type, check_validity=False, player='HUMAN')
                    solution[row][c] = move_type
            return True
        
        # Try each possible assignment for this row
        for assignment in all_row_assignments:
            if check_row_constraints_valid(row_idx, assignment):
                # Row is valid, record it
                for c in range(size):
                    solution[row_idx][c] = assignment[c]
                
                # Recurse to next row
                if dp_solve_row(row_idx + 1):
                    # Cache the full solution from this row onward
                    cached_solution = [row[:] for row in solution]
                    memo[memo_key] = cached_solution
                    return True
                
                # Undo this row's assignment
                undo_row(row_idx, size)
                for c in range(size):
                    solution[row_idx][c] = None
        
        # No valid assignment found for this row
        memo[memo_key] = None
        return False
    
    # Clear the grid first (in case partially filled)
    for r in range(size):
        for c in range(size):
            if game.grid[r][c] is not None:
                game.remove_move(r, c)
    
    # Reset state
    game.history = []
    game.scores = {'HUMAN': 0, 'CPU': 0}
    game.owners = [[None for _ in range(size)] for _ in range(size)]
    
    result = dp_solve_row(0)
    
    if result:
        # Verify the solution
        game.detect_cycle_dfs()
        game.check_completion()
    
    return result


# ==============================================================================
# MEMOIZED CONSTRAINT VALIDATOR
# ==============================================================================

class MemoizedValidator:
    """
    Wraps game validity checks with memoization to avoid redundant
    BFS traversals for cycle detection.
    
    DP Principle Applied:
    - Overlapping Subproblems: The same (cell, move_type, surrounding_state) 
      query occurs many times during backtracking/solving
    - Memoization: Cache results indexed by the local neighborhood state
    
    The cache key is based on the local neighborhood of the cell being checked,
    not the entire board state, making it memory-efficient while still useful.
    """
    
    def __init__(self, game):
        self.game = game
        self._cycle_cache = {}
        self._validity_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _get_local_state_key(self, r, c, move_type):
        """
        Generate a cache key based on the local neighborhood state.
        
        The key captures:
        - The cell position and proposed move
        - Degrees of the 4 corner nodes of the cell
        - Edges in the immediate neighborhood (adjacent cells)
        
        This is sufficient because:
        - Constraint checks only look at node degrees
        - Cycle checks depend on graph connectivity (captured by neighbor states)
        """
        size = self.game.size
        corners = [(r, c), (r+1, c), (r, c+1), (r+1, c+1)]
        
        # Node degrees of corners
        deg_state = tuple(self.game.node_degrees[n] for n in corners)
        
        # State of adjacent cells (up to 8 neighbors)
        neighbors = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    neighbors.append(self.game.grid[nr][nc])
                else:
                    neighbors.append('X')  # out of bounds marker
        
        return (r, c, move_type, deg_state, tuple(neighbors))
    
    def is_move_valid_cached(self, r, c, move_type, strict_cycles=True):
        """
        Cached version of game.is_move_valid().
        
        Uses memoization to skip redundant BFS cycle checks when the
        local neighborhood state hasn't changed.
        """
        key = self._get_local_state_key(r, c, move_type)
        
        if key in self._validity_cache:
            self.cache_hits += 1
            return self._validity_cache[key]
        
        self.cache_misses += 1
        result = self.game.is_move_valid(r, c, move_type, strict_cycles)
        self._validity_cache[key] = result
        return result
    
    def invalidate(self, r, c):
        """
        Invalidate cache entries affected by a change at (r, c).
        Called after a move is applied or undone.
        """
        size = self.game.size
        # Invalidate all cached entries in the neighborhood
        to_remove = []
        for key in self._validity_cache:
            kr, kc = key[0], key[1]
            if abs(kr - r) <= 1 and abs(kc - c) <= 1:
                to_remove.append(key)
        for key in to_remove:
            del self._validity_cache[key]
    
    def clear_cache(self):
        """Clear all cached entries."""
        self._validity_cache.clear()
        self._cycle_cache.clear()
    
    def get_stats(self):
        """Return cache performance statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total': total,
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size': len(self._validity_cache)
        }


# ==============================================================================
# DP-ENHANCED BACKTRACKING SOLVER
# ==============================================================================

def solve_with_dp_enhanced(game):
    """
    Enhanced backtracking solver that uses DP memoization for constraint validation.
    
    This combines the standard backtracking approach with:
    1. Memoized validity checks (avoids redundant BFS)
    2. Row-boundary state pruning (skips branches that can't satisfy constraints)
    
    This is more practical than pure Profile DP for larger boards while
    still demonstrating the DP paradigm through memoization.
    
    Args:
        game: SlantGame instance
    
    Returns:
        bool: True if solved, False if no solution
    """
    size = game.size
    validator = MemoizedValidator(game)
    
    # Row constraint completion cache
    # After filling all cells in rows 0..r, check if row r's top nodes are satisfied
    row_complete_cache = {}
    
    def are_row_nodes_satisfied(row_idx):
        """
        Check if all constrained nodes at the given row index 
        are fully satisfied (receiving no more edges).
        
        A node at row r is fully determined after all cells in rows r-1 and r are filled.
        """
        cache_key = tuple(
            game.node_degrees[(row_idx, c)] 
            for c in range(size + 1)
        )
        
        full_key = (row_idx, cache_key)
        if full_key in row_complete_cache:
            return row_complete_cache[full_key]
        
        result = True
        for c in range(size + 1):
            node = (row_idx, c)
            if node in game.constraints:
                if game.node_degrees[node] != game.constraints[node]:
                    result = False
                    break
        
        row_complete_cache[full_key] = result
        return result
    
    def backtrack(cell_idx):
        """
        Fill cells in row-major order, using DP memoization for validity checks
        and row-boundary pruning.
        """
        if cell_idx >= size * size:
            # All cells filled — verify ALL remaining constraints
            # Check the last row of nodes (row size-1) and bottom boundary (row size)
            for row_idx in range(size + 1):
                if not are_row_nodes_satisfied(row_idx):
                    return False
            # Final cycle check
            return not game.detect_cycle_dfs()
        
        r = cell_idx // size
        c = cell_idx % size
        
        # Row boundary pruning: when starting a new row (c == 0 and r > 0),
        # check if the previous row's BOUNDARY nodes are fully satisfied.
        # Node (r-1, c) is fully determined after filling rows r-2 and r-1.
        # This is a DP-style pruning: we prune early if intermediate
        # constraints are already violated.
        if c == 0 and r > 1:
            if not are_row_nodes_satisfied(r - 1):
                return False
        # Special case: after filling row 0, check row 0 nodes
        # (they only need edges from row 0)
        if c == 0 and r == 1:
            if not are_row_nodes_satisfied(0):
                return False
        
        # Skip already-filled cells
        if game.grid[r][c] is not None:
            return backtrack(cell_idx + 1)
        
        for move_type in ['L', 'R']:
            # Use memoized validity check
            if validator.is_move_valid_cached(r, c, move_type):
                game.apply_move(r, c, move_type, check_validity=False, player='HUMAN')
                validator.invalidate(r, c)
                
                if backtrack(cell_idx + 1):
                    return True
                
                game.undo()
                validator.invalidate(r, c)
        
        return False
    
    # Clear the grid
    for r in range(size):
        for c in range(size):
            if game.grid[r][c] is not None:
                game.remove_move(r, c)
    
    game.history = []
    game.scores = {'HUMAN': 0, 'CPU': 0}
    game.owners = [[None for _ in range(size)] for _ in range(size)]
    
    result = backtrack(0)
    
    if result:
        game.detect_cycle_dfs()
        game.check_completion()
    
    return result


# ==============================================================================
# SOLUTION EXTRACTION UTILITIES
# ==============================================================================

def extract_solution_grid(game):
    """
    Extract the solution grid from a solved game.
    
    Args:
        game: Solved SlantGame instance
    
    Returns:
        list: 2D grid of solution moves ('L' or 'R')
    """
    return [row[:] for row in game.grid]


def solve_and_extract(game, use_enhanced=True):
    """
    Solve the game using DP approach and extract solution grid.
    
    Args:
        game: SlantGame instance
        use_enhanced: If True, use DP-enhanced solver; else use profile DP
    
    Returns:
        tuple: (success: bool, solution_grid: list or None)
    """
    import copy
    
    # Create a copy to avoid modifying original
    game_copy = copy.deepcopy(game)
    
    if use_enhanced:
        success = solve_with_dp_enhanced(game_copy)
    else:
        success = solve_with_dp(game_copy)
    
    if success:
        return True, extract_solution_grid(game_copy)
    else:
        return False, None
