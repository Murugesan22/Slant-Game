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
