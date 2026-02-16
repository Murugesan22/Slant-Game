"""
Divide and Conquer Solver for Slant Game

Implements four D&C approaches:
1. Grid Partitioning Solver - Recursively divides grid into quadrants, solves each, merges
2. D&C Cycle Detection - Divides graph spatially, detects cycles in halves, checks cross-edges
3. Minimum-Boundary Cut Solver - Finds optimal split lines to decompose into independent sub-problems
4. Hybrid DP + D&C Solver - Uses DP boundary state memoization within a D&C framework

Algorithm Paradigm: Divide and Conquer
- Divide: Split the grid into smaller sub-grids (quadrants or halves)
- Conquer: Solve each sub-grid independently
- Combine: Resolve conflicts at shared boundary nodes

Time Complexity: O(2^(b) * T(n/4)) where b = boundary cells, T(n) = time for sub-problem of size n
Space Complexity: O(n * log(n)) for recursion stack
"""

from collections import deque


# ==============================================================================
# DIVIDE & CONQUER GRID SOLVER
# ==============================================================================

def solve_with_dnc(game):
    """
    Solve the Slant game using Divide and Conquer grid partitioning.
    
    Approach:
    1. DIVIDE: Split the grid into 4 quadrants (top-left, top-right, bottom-left, bottom-right)
    2. CONQUER: Recursively solve each quadrant independently
    3. COMBINE: Fix boundary conflicts where quadrants share nodes
    
    For a grid of size N:
    - Each quadrant is roughly (N/2) x (N/2)
    - Base case: 1x1 grid — try both L and R
    - Boundary nodes (shared between quadrants) need special handling during merge
    
    Args:
        game: SlantGame instance (modified in-place if solution found)
    
    Returns:
        bool: True if solved, False if no solution exists
    """
    size = game.size
    
    # Clear the grid first
    for r in range(size):
        for c in range(size):
            if game.grid[r][c] is not None:
                game.remove_move(r, c)
    
    game.history = []
    game.scores = {'HUMAN': 0, 'CPU': 0}
    game.owners = [[None for _ in range(size)] for _ in range(size)]
    
    # Build list of all cells to solve
    all_cells = [(r, c) for r in range(size) for c in range(size)]
    
    # Solve using D&C partitioning
    result = _dnc_solve(game, all_cells)
    
    if result:
        game.detect_cycle_dfs()
        game.check_completion()
    
    return result


def _dnc_solve(game, cells):
    """
    Divide & Conquer solver: partitions cells into quadrants, solves
    each quadrant, and combines results with backtracking at boundaries.
    
    Args:
        game: SlantGame instance
        cells: list of (r, c) tuples to solve
    
    Returns:
        bool: True if all cells successfully filled
    """
    if not cells:
        return True
    
    # BASE CASE: 1 cell — try both moves with backtracking
    if len(cells) == 1:
        r, c = cells[0]
        if game.grid[r][c] is not None:
            return True
        for move_type in ['L', 'R']:
            if game.is_move_valid(r, c, move_type):
                game.apply_move(r, c, move_type, check_validity=False, player='HUMAN')
                return True
        return False
    
    # BASE CASE: Small regions — use direct backtracking
    if len(cells) <= 4:
        return _backtrack_cells(game, cells, 0)
    
    # DIVIDE: Split into quadrants based on spatial midpoint
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    r_mid = (min(rows) + max(rows)) // 2
    c_mid = (min(cols) + max(cols)) // 2
    
    # Categorize cells into quadrants
    q1, q2, q3, q4 = [], [], [], []
    for r, c in cells:
        if r <= r_mid and c <= c_mid:
            q1.append((r, c))
        elif r <= r_mid:
            q2.append((r, c))
        elif c <= c_mid:
            q3.append((r, c))
        else:
            q4.append((r, c))
    
    quadrants = [q for q in [q1, q2, q3, q4] if q]
    
    # If splitting didn't help (all cells in one quadrant), use backtracking
    if len(quadrants) <= 1:
        return _backtrack_cells(game, cells, 0)
    
    # CONQUER + COMBINE: Solve quadrants sequentially with backtracking
    return _solve_quadrants_sequential(game, quadrants, 0)


def _solve_quadrants_sequential(game, quadrants, q_idx):
    """
    Solve quadrants one at a time. If a later quadrant fails,
    backtrack the current quadrant and try a different fill.
    
    This is the COMBINE step: each quadrant's solution must be
    compatible with all subsequent quadrants.
    """
    if q_idx >= len(quadrants):
        # All quadrants solved — verify full solution
        return _verify_full_solution(game)
    
    current_cells = quadrants[q_idx]
    
    # Try solving this quadrant using D&C recursion
    # We need to try ALL possible fills, not just the first one that works locally
    return _backtrack_with_continuation(game, current_cells, 0, quadrants, q_idx)


def _backtrack_with_continuation(game, cells, cell_idx, quadrants, q_idx):
    """
    Backtracking fill for the current quadrant's cells, with a continuation
    that attempts to solve remaining quadrants after this one is filled.
    
    This merges the CONQUER and COMBINE steps: we don't just find *any*
    valid fill for the current quadrant — we find one that's compatible
    with all remaining quadrants.
    """
    if cell_idx >= len(cells):
        # Current quadrant fully filled — continue to next quadrant
        return _solve_quadrants_sequential(game, quadrants, q_idx + 1)
    
    r, c = cells[cell_idx]
    
    # Skip already-filled cells
    if game.grid[r][c] is not None:
        return _backtrack_with_continuation(game, cells, cell_idx + 1, quadrants, q_idx)
    
    for move_type in ['L', 'R']:
        if game.is_move_valid(r, c, move_type):
            game.apply_move(r, c, move_type, check_validity=False, player='HUMAN')
            
            if _backtrack_with_continuation(game, cells, cell_idx + 1, quadrants, q_idx):
                return True
            
            game.undo()
    
    return False


def _backtrack_cells(game, cells, idx):
    """Simple backtracking for a small set of cells."""
    if idx >= len(cells):
        return True
    
    r, c = cells[idx]
    
    if game.grid[r][c] is not None:
        return _backtrack_cells(game, cells, idx + 1)
    
    for move_type in ['L', 'R']:
        if game.is_move_valid(r, c, move_type):
            game.apply_move(r, c, move_type, check_validity=False, player='HUMAN')
            
            if _backtrack_cells(game, cells, idx + 1):
                return True
            
            game.undo()
    
    return False


def _verify_full_solution(game):
    """Verify the complete grid is a valid solution."""
    size = game.size
    
    # All cells must be filled
    for r in range(size):
        for c in range(size):
            if game.grid[r][c] is None:
                return False
    
    # All constraints must be satisfied
    for node, limit in game.constraints.items():
        deg = game.node_degrees[node]
        if deg != limit:
            return False
    
    # No cycles
    return not game.detect_cycle_dfs()


# ==============================================================================
# DIVIDE & CONQUER CYCLE DETECTION
# ==============================================================================

def detect_cycle_dnc(game):
    """
    Detect cycles in the Slant game graph using Divide and Conquer.
    
    Approach:
    1. DIVIDE: Split graph nodes into two halves based on spatial position
       (left half and right half based on column midpoint)
    2. CONQUER: Recursively detect cycles within each half
    3. COMBINE: Check for cycles that span both halves by examining
       cross-partition edges
    
    This is a spatial decomposition of the standard cycle detection problem.
    
    Args:
        game: SlantGame instance
    
    Returns:
        bool: True if a cycle is detected, list of loop cells
    """
    graph = game.get_graph_representation()
    all_nodes = [node for node in graph if len(graph[node]) > 0]
    
    if not all_nodes:
        return False, []
    
    loop_cells = []
    has_cycle = _dnc_cycle_detect(graph, all_nodes, game, loop_cells)
    
    return has_cycle, loop_cells


def _dnc_cycle_detect(graph, nodes, game, loop_cells):
    """
    Recursive D&C cycle detection.
    
    DIVIDE: Split nodes into left/right halves by column coordinate
    CONQUER: Detect cycles in each half
    COMBINE: Check for cycles through cross-boundary edges
    """
    if len(nodes) <= 2:
        # Base case: 2 or fewer nodes — check for multi-edges
        if len(nodes) == 2:
            n1, n2 = nodes
            # Check if there are multiple edges between these nodes
            edge_count = graph[n1].count(n2)
            if edge_count > 1:
                return True
        return False
    
    # DIVIDE: Split by column coordinate
    nodes_sorted = sorted(nodes, key=lambda n: (n[1], n[0]))
    mid = len(nodes_sorted) // 2
    left_nodes = set(nodes_sorted[:mid])
    right_nodes = set(nodes_sorted[mid:])
    
    # CONQUER: Check each half for internal cycles
    
    # Left half: subgraph induced by left_nodes
    if _has_cycle_in_subgraph(graph, left_nodes):
        _find_cycle_cells(graph, left_nodes, game, loop_cells)
        return True
    
    # Right half: subgraph induced by right_nodes
    if _has_cycle_in_subgraph(graph, right_nodes):
        _find_cycle_cells(graph, right_nodes, game, loop_cells)
        return True
    
    # COMBINE: Check for cycles that cross the partition
    # A cross-partition cycle uses edges from both halves.
    # We detect this by:
    # 1. Find all cross-edges (edges between left and right nodes)
    # 2. For each cross-edge pair, check if adding them creates connectivity
    #    that forms a cycle
    
    cross_edges = []
    for node in left_nodes:
        for neighbor in graph[node]:
            if neighbor in right_nodes:
                cross_edges.append((node, neighbor))
    
    if len(cross_edges) < 2:
        return False  # Need at least 2 cross-edges for a cross-partition cycle
    
    # Build a contracted graph:
    # - Find connected components in left half and right half
    # - Each component becomes a super-node
    # - Cross-edges connect super-nodes
    # - A cycle in the contracted graph means a cross-partition cycle exists
    
    left_components = _find_components(graph, left_nodes)
    right_components = _find_components(graph, right_nodes)
    
    # Map nodes to component IDs
    node_to_comp = {}
    for i, comp in enumerate(left_components):
        for node in comp:
            node_to_comp[node] = ('L', i)
    for i, comp in enumerate(right_components):
        for node in comp:
            node_to_comp[node] = ('R', i)
    
    # Build contracted graph
    contracted = {}
    for u, v in cross_edges:
        cu = node_to_comp.get(u)
        cv = node_to_comp.get(v)
        if cu and cv:
            if cu not in contracted:
                contracted[cu] = []
            if cv not in contracted:
                contracted[cv] = []
            contracted[cu].append(cv)
            contracted[cv].append(cu)
    
    # Check for cycle in contracted graph (simple DFS)
    if _has_cycle_in_contracted(contracted):
        # Find the actual cycle cells
        _find_cross_cycle_cells(graph, left_nodes, right_nodes, 
                                cross_edges, game, loop_cells)
        return True
    
    return False


def _has_cycle_in_subgraph(graph, node_set):
    """Check if the subgraph induced by node_set contains a cycle using DFS."""
    visited = set()
    
    for start in node_set:
        if start in visited:
            continue
        
        # DFS with parent tracking
        stack = [(start, None)]
        visited.add(start)
        
        while stack:
            node, parent = stack.pop()
            
            for neighbor in graph[node]:
                if neighbor not in node_set:
                    continue  # Skip edges outside this partition
                
                if neighbor == parent:
                    continue
                
                if neighbor in visited:
                    return True  # Cycle found
                
                visited.add(neighbor)
                stack.append((neighbor, node))
    
    return False


def _find_components(graph, node_set):
    """Find connected components within a subgraph induced by node_set."""
    visited = set()
    components = []
    
    for start in node_set:
        if start in visited:
            continue
        
        component = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            node = queue.popleft()
            component.add(node)
            
            for neighbor in graph[node]:
                if neighbor in node_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        components.append(component)
    
    return components


def _has_cycle_in_contracted(contracted):
    """Check for cycle in the contracted/super-node graph."""
    if not contracted:
        return False
    
    visited = set()
    
    for start in contracted:
        if start in visited:
            continue
        
        stack = [(start, None)]
        visited.add(start)
        
        while stack:
            node, parent = stack.pop()
            
            for neighbor in contracted.get(node, []):
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    return True
                visited.add(neighbor)
                stack.append((neighbor, node))
    
    return False


def _find_cycle_cells(graph, node_set, game, loop_cells):
    """Find cells corresponding to cycle edges within a node set."""
    visited = set()
    
    for start in node_set:
        if start in visited:
            continue
        
        stack = [(start, None, [start])]
        visited.add(start)
        
        while stack:
            node, parent, path = stack.pop()
            
            for neighbor in graph[node]:
                if neighbor not in node_set:
                    continue
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    # Found cycle — extract cells
                    if neighbor in path:
                        cycle_start = path.index(neighbor)
                        cycle_path = path[cycle_start:]
                        for i in range(len(cycle_path)):
                            u = cycle_path[i]
                            v = cycle_path[(i + 1) % len(cycle_path)]
                            _add_edge_to_cells(u, v, game, loop_cells)
                    return
                
                visited.add(neighbor)
                stack.append((neighbor, node, path + [neighbor]))


def _find_cross_cycle_cells(graph, left_nodes, right_nodes, 
                             cross_edges, game, loop_cells):
    """Find cells involved in a cross-partition cycle."""
    # Use BFS to find a cycle path through cross-edges
    for u, v in cross_edges:
        # Try to find a path from v back to u using the graph
        visited = {v}
        queue = deque([(v, [v])])
        
        while queue:
            node, path = queue.popleft()
            
            for neighbor in graph[node]:
                if neighbor == u and len(path) > 1:
                    # Found cycle
                    full_path = [u] + path
                    for i in range(len(full_path)):
                        n1 = full_path[i]
                        n2 = full_path[(i + 1) % len(full_path)]
                        _add_edge_to_cells(n1, n2, game, loop_cells)
                    return
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))


def _add_edge_to_cells(u, v, game, loop_cells):
    """Convert an edge (u, v) to its corresponding grid cell."""
    r1, c1 = u
    r2, c2 = v
    min_r, min_c = min(r1, r2), min(c1, c2)
    
    if 0 <= min_r < game.size and 0 <= min_c < game.size:
        cell = (min_r, min_c)
        if cell not in loop_cells:
            loop_cells.append(cell)


# ==============================================================================
# D&C-BASED MOVE EVALUATION
# ==============================================================================

def evaluate_move_dnc(game, r, c, move_type):
    """
    Evaluate a move using D&C quadrant analysis.
    
    Divides the board into quadrants and scores the move based on:
    1. Which quadrant it belongs to
    2. How constrained that quadrant is
    3. Whether it's a boundary cell (connecting quadrants)
    4. Impact on the quadrant's solvability
    
    Args:
        game: SlantGame instance
        r, c: Cell position
        move_type: 'L' or 'R'
    
    Returns:
        float: Score for this move
    """
    size = game.size
    mid_r = size // 2
    mid_c = size // 2
    
    score = 1.0
    
    # Determine which quadrant this cell belongs to
    if r <= mid_r and c <= mid_c:
        quadrant = 'TL'
    elif r <= mid_r and c > mid_c:
        quadrant = 'TR'
    elif r > mid_r and c <= mid_c:
        quadrant = 'BL'
    else:
        quadrant = 'BR'
    
    # Calculate constraint density per quadrant
    quadrant_constraints = {'TL': 0, 'TR': 0, 'BL': 0, 'BR': 0}
    quadrant_unsatisfied = {'TL': 0, 'TR': 0, 'BL': 0, 'BR': 0}
    
    for node, limit in game.constraints.items():
        nr, nc = node
        if nr <= mid_r + 1 and nc <= mid_c + 1:
            quadrant_constraints['TL'] += 1
            if game.node_degrees[node] != limit:
                quadrant_unsatisfied['TL'] += 1
        if nr <= mid_r + 1 and nc >= mid_c:
            quadrant_constraints['TR'] += 1
            if game.node_degrees[node] != limit:
                quadrant_unsatisfied['TR'] += 1
        if nr >= mid_r and nc <= mid_c + 1:
            quadrant_constraints['BL'] += 1
            if game.node_degrees[node] != limit:
                quadrant_unsatisfied['BL'] += 1
        if nr >= mid_r and nc >= mid_c:
            quadrant_constraints['BR'] += 1
            if game.node_degrees[node] != limit:
                quadrant_unsatisfied['BR'] += 1
    
    # Bonus for moves in the most constrained quadrant
    max_unsatisfied = max(quadrant_unsatisfied.values()) if quadrant_unsatisfied else 0
    if quadrant_unsatisfied[quadrant] == max_unsatisfied and max_unsatisfied > 0:
        score += 0.4  # Prioritize most constrained quadrant
    
    # Boundary bonus: cells at quadrant boundaries are critical for merge
    is_boundary = (r == mid_r or r == mid_r + 1 or c == mid_c or c == mid_c + 1)
    if is_boundary:
        score += 0.3  # Boundary cells are important for D&C merge
    
    # Constraint satisfaction check
    if move_type == 'L':
        nodes = [(r, c), (r + 1, c + 1)]
    else:
        nodes = [(r + 1, c), (r, c + 1)]
    
    for node in nodes:
        current_deg = game.node_degrees[node]
        limit = game.constraints.get(node)
        
        if limit is not None:
            new_deg = current_deg + 1
            if new_deg == limit:
                score += 0.5  # Satisfies constraint
            elif new_deg < limit:
                score += 0.2  # Progress toward constraint
            else:
                score -= 100  # Violation
        else:
            if current_deg + 1 > 4:
                score -= 100
    
    # Line continuity bonus
    for node in nodes:
        if game.node_degrees[node] > 0:
            score += 0.1
    
    return score




# ==============================================================================
# MINIMUM-BOUNDARY CUT SOLVER
# ==============================================================================

def solve_with_boundary_cut(game):
    """
    Solve the Slant game using Minimum-Boundary Cut D&C.
    
    Strategy:
    - Scan all possible horizontal/vertical cut lines through the grid
    - Score each cut line by its "freedom" (how many unconstrained possibilities exist)
    - Cut along the line with the LOWEST freedom score (most determined)
    - Solve the cut line cells first, then solve the two independent halves
    
    This avoids the "Border Problem" entirely when a perfect cut exists,
    because cut lines with 0 freedom are fully determined walls.
    
    Args:
        game: SlantGame instance (modified in-place if solution found)
    
    Returns:
        bool: True if solved, False if no solution exists
    """
    size = game.size
    
    # Clear the grid first
    for r in range(size):
        for c in range(size):
            if game.grid[r][c] is not None:
                game.remove_move(r, c)
    
    game.history = []
    game.scores = {'HUMAN': 0, 'CPU': 0}
    game.owners = [[None for _ in range(size)] for _ in range(size)]
    
    # Build list of all cells to solve
    all_cells = [(r, c) for r in range(size) for c in range(size)]
    
    # Solve using boundary cut decomposition
    result = _boundary_cut_solve(game, all_cells, 0)
    
    if result:
        game.detect_cycle_dfs()
        game.check_completion()
    
    return result


def _calc_line_freedom(game, nodes):
    """
    Calculate the "freedom score" for a line of nodes.
    
    Nodes with constraint 0 or 4 have 0 freedom (fully determined).
    Nodes at grid corners have limited freedom.
    Unconstrained nodes have maximum freedom.
    
    Lower score = better cut (more determined, less ambiguity).
    """
    score = 0
    for node in nodes:
        constraint = game.constraints.get(node, -1)
        current_deg = game.node_degrees.get(node, 0)
        
        if constraint == 0:
            # Node must have 0 connections — fully blocked
            score += 0
        elif constraint == 4:
            # Node must have 4 connections — fully forced
            score += 0
        elif constraint >= 0:
            # Partially constrained — some freedom remains
            remaining = constraint - current_deg
            score += max(0, remaining)
        else:
            # Unconstrained node — maximum freedom
            # Calculate how many more connections are possible (max 4)
            score += max(0, 4 - current_deg)
    
    return score


def _find_best_cut(game, cells):
    """
    Find the optimal cut line for the given set of cells.
    
    Evaluates all possible horizontal and vertical cuts through the region
    and returns the one with the lowest freedom score.
    
    Returns:
        (orientation, position, score) or None if no useful cut exists.
        orientation: 'H' for horizontal, 'V' for vertical
        position: the row/column index to cut at
    """
    if not cells:
        return None
    
    size = game.size
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    
    best_cut = None
    best_score = float('inf')
    
    # Evaluate horizontal cuts (rows of nodes between cell rows)
    # A horizontal cut at row_idx means the node row at row_idx+1
    for cut_r in range(min_r, max_r + 1):
        # The node row between cell rows cut_r and cut_r+1
        node_row = cut_r + 1
        if node_row <= 0 or node_row >= size:
            continue
        
        # Nodes along this horizontal line within our column range
        nodes = [(node_row, c) for c in range(min_c, max_c + 2)]
        nodes = [n for n in nodes if 0 <= n[0] <= size and 0 <= n[1] <= size]
        
        if not nodes:
            continue
        
        freedom = _calc_line_freedom(game, nodes)
        if freedom < best_score:
            best_score = freedom
            best_cut = ('H', cut_r, freedom)
    
    # Evaluate vertical cuts (columns of nodes between cell columns)
    for cut_c in range(min_c, max_c + 1):
        node_col = cut_c + 1
        if node_col <= 0 or node_col >= size:
            continue
        
        nodes = [(r, node_col) for r in range(min_r, max_r + 2)]
        nodes = [n for n in nodes if 0 <= n[0] <= size and 0 <= n[1] <= size]
        
        if not nodes:
            continue
        
        freedom = _calc_line_freedom(game, nodes)
        if freedom < best_score:
            best_score = freedom
            best_cut = ('V', cut_c, freedom)
    
    return best_cut


def _boundary_cut_solve(game, cells, depth):
    """
    Recursive boundary-cut D&C solver.
    
    1. Find the best cut line through the region
    2. Solve cells touching the cut line first (small set)
    3. Split remaining cells into two independent halves
    4. Recursively solve each half
    
    Falls back to backtracking for small regions or when no good cut exists.
    """
    if not cells:
        return True
    
    # BASE CASE: Small region — direct backtracking
    if len(cells) <= 4:
        return _backtrack_cells(game, cells, 0)
    
    # Prevent excessive recursion depth
    if depth > 20:
        return _backtrack_cells(game, cells, 0)
    
    # DIVIDE: Find optimal cut
    cut = _find_best_cut(game, cells)
    
    # If no useful cut found, fall back to standard D&C
    if cut is None:
        return _backtrack_cells(game, cells, 0)
    
    orientation, position, freedom = cut
    
    # If the cut line has very high freedom (> threshold), backtrack instead
    # High freedom means the cut won't help much
    if freedom > len(cells) * 2:
        return _dnc_solve(game, cells)
    
    # SPLIT cells into: cut_adjacent (touching the cut) and two halves
    half_a = []
    half_b = []
    cut_adjacent = []
    
    if orientation == 'H':
        # Horizontal cut at row `position` — node row is position+1
        for r, c in cells:
            if r == position or r == position + 1:
                # Cell touches the cut line
                cut_adjacent.append((r, c))
            elif r < position:
                half_a.append((r, c))
            else:
                half_b.append((r, c))
    else:
        # Vertical cut at column `position` — node column is position+1
        for r, c in cells:
            if c == position or c == position + 1:
                cut_adjacent.append((r, c))
            elif c < position:
                half_a.append((r, c))
            else:
                half_b.append((r, c))
    
    # If splitting didn't produce two real halves, just backtrack
    if not half_a or not half_b:
        return _dnc_solve(game, cells)
    
    # CONQUER: Solve cut-adjacent cells first (bridges between halves)
    # Then solve each half independently
    return _solve_cut_then_halves(game, cut_adjacent, half_a, half_b, 0, depth)


def _solve_cut_then_halves(game, cut_cells, half_a, half_b, cut_idx, depth):
    """
    First backtrack through cut-adjacent cells, then solve each half independently.
    
    This ensures the "bridge" between halves is locked before solving either side,
    eliminating the border problem for the independent halves.
    """
    if cut_idx >= len(cut_cells):
        # All cut cells filled — now solve halves independently
        if not _boundary_cut_solve(game, half_a, depth + 1):
            return False
        if not _boundary_cut_solve(game, half_b, depth + 1):
            # Undo half_a and fail
            for r, c in half_a:
                if game.grid[r][c] is not None:
                    game.remove_move(r, c)
            return False
        return True
    
    r, c = cut_cells[cut_idx]
    
    # Skip already-filled cells
    if game.grid[r][c] is not None:
        return _solve_cut_then_halves(game, cut_cells, half_a, half_b, cut_idx + 1, depth)
    
    for move_type in ['L', 'R']:
        if game.is_move_valid(r, c, move_type):
            game.apply_move(r, c, move_type, check_validity=False, player='HUMAN')
            
            if _solve_cut_then_halves(game, cut_cells, half_a, half_b, cut_idx + 1, depth):
                return True
            
            game.undo()
    
    return False


# ==============================================================================
# HYBRID DP + D&C SOLVER
# ==============================================================================

def solve_with_hybrid(game):
    """
    Solve the Slant game using a Hybrid DP + D&C approach.
    
    Strategy:
    - DIVIDE: Split the grid into two halves (top/bottom or left/right)
    - CONQUER: For each half, use DP row-by-row to compute the set of
      valid boundary states (the "menu" of possible outputs)
    - COMBINE: Find the intersection of compatible boundary states
      between the two halves and pick a matching state
    
    This transforms the border problem from trial-and-error backtracking
    into a set intersection problem.
    
    Falls back to DP-enhanced backtracking if the hybrid approach fails.
    
    Args:
        game: SlantGame instance (modified in-place if solution found)
    
    Returns:
        bool: True if solved, False if no solution exists
    """
    size = game.size
    
    # Clear the grid first
    for r in range(size):
        for c in range(size):
            if game.grid[r][c] is not None:
                game.remove_move(r, c)
    
    game.history = []
    game.scores = {'HUMAN': 0, 'CPU': 0}
    game.owners = [[None for _ in range(size)] for _ in range(size)]
    
    # DIVIDE: Split into top half and bottom half
    mid_row = size // 2
    
    # CONQUER: Use DP to solve top half and collect valid boundary states
    top_states = _collect_boundary_states_dp(game, 0, mid_row)
    
    if not top_states:
        # No valid solutions for top half — fall back to standard D&C
        return _dnc_solve(game, [(r, c) for r in range(size) for c in range(size)])
    
    # COMBINE: Try each top-half boundary state and see if bottom half works
    for boundary_state, top_assignments in top_states.items():
        # Apply top half assignments
        _apply_assignments(game, top_assignments, 0, mid_row)
        
        # Try to solve the bottom half with this boundary committed
        bottom_cells = [(r, c) for r in range(mid_row, size) for c in range(size)]
        
        if _backtrack_cells(game, bottom_cells, 0):
            # Verify the full solution
            if _verify_full_solution(game):
                game.detect_cycle_dfs()
                game.check_completion()
                return True
        
        # Undo bottom half
        for r in range(mid_row, size):
            for c in range(size):
                if game.grid[r][c] is not None:
                    game.remove_move(r, c)
        
        # Undo top half
        for r in range(mid_row):
            for c in range(size):
                if game.grid[r][c] is not None:
                    game.remove_move(r, c)
    
    # Fallback: Use standard D&C if hybrid approach exhausted all options
    return _dnc_solve(game, [(r, c) for r in range(size) for c in range(size)])


def _get_boundary_state_at_row(game, row_idx):
    """
    Capture the boundary state at a given row of nodes.
    
    The boundary state is a tuple of (degree, constraint) pairs for each
    node along the specified row. This serves as the memoization key.
    """
    size = game.size
    state = []
    for c in range(size + 1):
        node = (row_idx, c)
        deg = game.node_degrees.get(node, 0)
        constraint = game.constraints.get(node, -1)
        state.append((deg, constraint))
    return tuple(state)


def _collect_boundary_states_dp(game, start_row, end_row):
    """
    Use DP to enumerate valid assignments for rows [start_row, end_row)
    and collect the resulting boundary states at end_row.
    
    Returns:
        dict: {boundary_state: assignment_list} mapping boundary states
              to the row assignments that produce them.
              Limited to MAX_STATES to prevent memory explosion.
    """
    import itertools
    
    size = game.size
    num_rows = end_row - start_row
    
    if num_rows <= 0:
        return {}
    
    MAX_STATES = 50  # Limit to prevent memory explosion on large grids
    boundary_states = {}
    
    # Generate all possible row assignments (each cell is L or R)
    # For efficiency, we process row by row with pruning
    
    def dp_collect(row_idx, assignments_so_far):
        """Recursively process rows, pruning invalid assignments."""
        if len(boundary_states) >= MAX_STATES:
            return  # Hit limit
        
        if row_idx >= end_row:
            # All rows assigned — capture the boundary state
            boundary = _get_boundary_state_at_row(game, end_row)
            if boundary not in boundary_states:
                boundary_states[boundary] = list(assignments_so_far)
            return
        
        # Try all 2^size assignments for this row
        # For small sizes this is feasible; for large sizes we limit exploration
        max_assignments = min(2 ** size, 128)  # Cap at 128 to prevent explosion
        
        for bits in range(max_assignments):
            if len(boundary_states) >= MAX_STATES:
                return
            
            assignment = []
            valid = True
            
            # Build assignment from bits
            for c in range(size):
                move_type = 'L' if (bits >> c) & 1 == 0 else 'R'
                assignment.append((row_idx, c, move_type))
            
            # Apply the assignment
            applied = []
            for r, c, mt in assignment:
                if game.grid[r][c] is not None:
                    # Cell already filled (shouldn't happen in clean solve)
                    applied.append(False)
                    continue
                if game.is_move_valid(r, c, mt):
                    game.apply_move(r, c, mt, check_validity=False, player='HUMAN')
                    applied.append(True)
                else:
                    valid = False
                    break
            
            if valid and len(applied) == size and all(
                game.grid[row_idx][c] is not None for c in range(size)
            ):
                # Check row constraints (nodes at row_idx are now fully determined
                # if row_idx > start_row)
                row_ok = True
                if row_idx > start_row:
                    for c in range(size + 1):
                        node = (row_idx, c)
                        constraint = game.constraints.get(node, -1)
                        if constraint >= 0:
                            deg = game.node_degrees.get(node, 0)
                            if deg > constraint:
                                row_ok = False
                                break
                
                if row_ok:
                    dp_collect(row_idx + 1, assignments_so_far + assignment)
            
            # Undo the assignment
            for i in range(len(applied) - 1, -1, -1):
                if applied[i]:
                    game.undo()
    
    dp_collect(start_row, [])
    
    return boundary_states


def _apply_assignments(game, assignments, start_row, end_row):
    """
    Apply a list of (row, col, move_type) assignments to the game.
    """
    for r, c, mt in assignments:
        if start_row <= r < end_row and game.grid[r][c] is None:
            if game.is_move_valid(r, c, mt):
                game.apply_move(r, c, mt, check_validity=False, player='HUMAN')


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


def solve_and_extract_dnc(game):
    """
    Solve the game using D&C approach and extract solution grid.
    
    Args:
        game: SlantGame instance
    
    Returns:
        tuple: (success: bool, solution_grid: list or None)
    """
    import copy
    
    # Create a copy to avoid modifying original
    game_copy = copy.deepcopy(game)
    
    success = solve_with_dnc(game_copy)
    
    if success:
        return True, extract_solution_grid(game_copy)
    else:
        return False, None


def solve_and_extract_hybrid(game):
    """
    Solve the game using Hybrid DP+D&C approach and extract solution grid.
    
    Args:
        game: SlantGame instance
    
    Returns:
        tuple: (success: bool, solution_grid: list or None)
    """
    import copy
    
    # Create a copy to avoid modifying original
    game_copy = copy.deepcopy(game)
    
    success = solve_with_hybrid(game_copy)
    
    if success:
        return True, extract_solution_grid(game_copy)
    else:
        return False, None


def solve_and_extract_cut(game):
    """
    Solve the game using Boundary Cut approach and extract solution grid.
    
    Args:
        game: SlantGame instance
    
    Returns:
        tuple: (success: bool, solution_grid: list or None)
    """
    import copy
    
    # Create a copy to avoid modifying original
    game_copy = copy.deepcopy(game)
    
    success = solve_with_boundary_cut(game_copy)
    
    if success:
        return True, extract_solution_grid(game_copy)
    else:
        return False, None

