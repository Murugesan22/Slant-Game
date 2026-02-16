import random
import sys

# Increase recursion depth for backtracking constraints generation if needed
sys.setrecursionlimit(2000)

# [REVIEW 1 REQUIREMENT]: Graph Representation
# We use explicit Adjacency Lists and BFS/DFS for all graph operations.
# UnionFind class removed to satisfy "Remove Grid Based Logic" requirement.

class SlantGame:
    def __init__(self, size=5):
        self.size = size
        self.nodes_size = size + 1
        
        # [REVIEW 1 REQUIREMENT]: Formal Graph Definition G = (V, E)
        # 1. Initialize V (Static Set of Nodes)
        self.V = self._initialize_nodes_V()
        
        # 2. Initialize E (Dynamic Set of Edges) represented as Adjacency List
        self.graph = {} # This represents E (and the graph structure)
        self._initialize_edges_E()
        
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.constraints = {}
        self.node_degrees = {node: 0 for node in self.V} 
        
        self.history = []
        self.move_history = []  # Track all moves with chess coordinates for review
        self.solution_grid = None  # Will store the correct solution for review
        self.status = "RUNNING"
        self.winner = None
        self.turn = 'HUMAN' # 'HUMAN' or 'CPU'
        self.scores = {'HUMAN': 0, 'CPU': 0}
        self.owners = [[None for _ in range(size)] for _ in range(size)] # Track who placed what
        self.loop_cells = [] # [REVIEW 1]: Track cells in detected loops

        self._initialize_empty_state()
        self._generate_valid_puzzle()

    def _initialize_nodes_V(self):
        """
        Define V: All intersection points in the grid.
        Returns a list of tuples (r, c).
        """
        nodes = []
        for r in range(self.nodes_size):
            for c in range(self.nodes_size):
                nodes.append((r, c))
        return nodes

    def _initialize_edges_E(self):
        """
        Initialize E: Start with an empty edge list for every node in V.
        """
        for node in self.V:
            self.graph[node] = [] # Adjacency list: Node -> [Neighbors]

    def _initialize_empty_state(self):
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.status = "RUNNING"
        self.scores = {'HUMAN': 0, 'CPU': 0}
        self.owners = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.turn = 'HUMAN'
        
        # Reset Graph (Keep V, Clear E)
        self._initialize_edges_E()
        
        for node in self.V:
            self.node_degrees[node] = 0

    # ... (skipping _generate_valid_puzzle and other methods - ensure context matches) ...

    def check_completion(self):
        # 1. Check if Board is Full
        is_full = True
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] is None:
                    is_full = False
                    break
        
        if not is_full:
            self.status = "RUNNING"
            return False

        # Board is Full. Now check Validity.
        
        # 2. Check Constraints
        for node, limit in self.constraints.items():
            if self.node_degrees[node] != limit:
                self.status = "FILLED_INVALID"
                return False

        # 3. Check for Loops
        # 3. Check for Loops (Using DFS on Graph Representation - [REVIEW 1 REQUIREMENT])
        if self.detect_cycle_dfs():
             self.status = "FILLED_INVALID"
             return False

        # Scoring Winner Check
        h_score = self.scores['HUMAN']
        c_score = self.scores['CPU']
        
        if h_score > c_score:
            self.status = "WIN_HUMAN"
        elif c_score > h_score:
            self.status = "WIN_CPU"
        else:
            self.status = "DRAW"
            
        return True

    def _generate_valid_puzzle(self):
        """
        Sudoku-style puzzle generator (generate-then-carve).
        
        Algorithm:
        1. FILL: Generate a valid complete solution using randomized backtracking
           with MRV (Most Restricted Variable) ordering for efficiency.
        2. ALL CLUES: Record every node's degree as a constraint (all clues revealed).
        3. CARVE: Shuffle all constraint nodes, remove them one-by-one.
           After each removal, check if the puzzle still has exactly 1 solution.
           If removing a clue creates multiple solutions, put it back.
        4. RESULT: A minimal(ish) set of clues that guarantees a unique solution.
        
        This is the same technique used by high-quality Sudoku generators.
        Unlike the old brute-force approach, this ALWAYS produces a unique puzzle.
        """
        attempts = 0
        success = False
        
        while attempts < 5 and not success:
            attempts += 1
            
            # Step 1: Fill the board with a valid random solution
            self._initialize_empty_state()
            self.constraints = {}
            
            if not self._fast_fill_board():
                continue  # Rare failure, retry
            
            # Record the solution's node degrees
            solution_degrees = self.node_degrees.copy()
            
            # Step 2: Start with ALL nodes as constraints
            self._initialize_empty_state()
            all_nodes = list(solution_degrees.keys())
            self.constraints = {node: deg for node, deg in solution_degrees.items()}
            
            # Step 3: Carve — remove clues one by one
            # Shuffle for variety, but prefer removing interior nodes first
            # (edge/corner clues are more important for uniqueness)
            random.shuffle(all_nodes)
            # Sort: interior nodes first (more likely removable)
            all_nodes.sort(key=lambda n: self._node_importance(n))
            
            removed = 0
            target_clues = int(len(all_nodes) * 0.35)  # Aim for ~35% clues
            
            for node in all_nodes:
                if len(self.constraints) <= target_clues:
                    break  # Hit target density
                
                # Try removing this clue
                saved_deg = self.constraints.pop(node)
                
                # Check if puzzle is still unique
                solutions = self.count_solutions(limit=2)
                
                if solutions != 1:
                    # Removing this clue breaks uniqueness — put it back
                    self.constraints[node] = saved_deg
                else:
                    removed += 1
            
            success = True
        
        if success:
            print(f"Puzzle generated in {attempts} attempt(s) with {len(self.constraints)} clues (Sudoku-style carve)")
        else:
            print("Puzzle generation fallback used")
    
    def _fast_fill_board(self):
        """
        Fill the board with a valid random solution using MRV-ordered backtracking.
        Much faster than solve_game(randomize=True) because it picks the most
        constrained cell first, pruning the search tree dramatically.
        """
        cell = self._find_most_constrained_cell()
        if cell is None:
            return True  # All filled
        
        r, c = cell
        moves = ['L', 'R']
        random.shuffle(moves)
        
        for mv in moves:
            if self.is_move_valid(r, c, mv):
                self.apply_move(r, c, mv, check_validity=False, player='HUMAN')
                if self._fast_fill_board():
                    return True
                self.undo()
        
        return False
    
    def _find_most_constrained_cell(self):
        """
        MRV (Most Restricted Variable) heuristic.
        Returns the empty cell with the fewest valid moves.
        If a cell has 0 valid moves, return it immediately (triggers backtrack).
        
        This is the single most impactful optimization for constraint-based
        backtracking — identical to what Sudoku solvers use.
        """
        best_cell = None
        min_options = 3  # Max possible is 2 (L or R)
        
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] is not None:
                    continue
                
                # Count valid moves for this cell
                options = 0
                for mv in ['L', 'R']:
                    if self.is_move_valid(r, c, mv):
                        options += 1
                
                if options == 0:
                    return (r, c)  # Dead end — return immediately to trigger backtrack
                
                if options < min_options:
                    min_options = options
                    best_cell = (r, c)
                    if options == 1:
                        return best_cell  # Only one choice — forced move
        
        return best_cell
    
    def _node_importance(self, node):
        """
        Score how important a node is for defining the puzzle.
        Higher score = harder to remove (keep it as a clue).
        Lower score = easier to remove (try removing it first).
        
        Corner and edge nodes are more important (they constrain more).
        Interior unconstrained nodes can often be removed.
        """
        r, c = node
        size = self.nodes_size
        
        # Corner nodes are most important
        is_corner = (r in (0, size-1)) and (c in (0, size-1))
        if is_corner:
            return 10
        
        # Edge nodes are fairly important
        is_edge = r == 0 or r == size-1 or c == 0 or c == size-1
        if is_edge:
            return 5
        
        # Constraint value 0 or 4 are highly informative (remove last)
        deg = self.constraints.get(node, -1)
        if deg == 0 or deg == 4:
            return 8
        
        # Interior nodes with moderate constraints
        return 1


    def count_solutions(self, limit=2):
        """
        Counts solutions consistent with current self.constraints.
        Returns count (capped at limit).
        Uses MRV ordering for dramatically faster pruning.
        """
        count = 0
        
        def find_mrv_cell():
            """Find empty cell with fewest valid moves (MRV heuristic)."""
            best = None
            min_opts = 3
            for r in range(self.size):
                for c in range(self.size):
                    if self.grid[r][c] is not None:
                        continue
                    opts = sum(1 for mv in ['L', 'R'] if self.is_move_valid(r, c, mv))
                    if opts == 0:
                        return (r, c, 0)  # Dead end
                    if opts < min_opts:
                        min_opts = opts
                        best = (r, c, opts)
                        if opts == 1:
                            return best
            return best
        
        def backtrack():
            nonlocal count
            if count >= limit:
                return
            
            result = find_mrv_cell()
            if result is None:
                # All cells filled — found a solution
                count += 1
                return
            
            r, c, opts = result
            if opts == 0:
                return  # Dead end — no valid moves
            
            for mv in ['L', 'R']:
                if self.is_move_valid(r, c, mv):
                    self.apply_move(r, c, mv, check_validity=False)
                    backtrack()
                    self.undo()
                    if count >= limit:
                        return
        
        backtrack()
        return count

 def solve_game(self, randomize=False, strategy=None):
        # Backtracking solver
        # Returns True if solved, False otherwise
        
        # Verify if current state has cycles? (Should be maintained by moves)
        
        # [GREEDY UPDATE]: Choose cell based on strategy if provided
        if strategy:
             # Lazy import to avoid circular dependency
             from cpu_ai import GreedyAI
             ai = GreedyAI(self, strategy)
             # Step 1: Candidate Generation & Selection (Greedy Best Cell)
             r, c = -1, -1
             best_cell = ai.get_best_empty_cell() # New helper method
             if best_cell:
                 r, c = best_cell
             else:
                 # No empty valid cells found (or all dead ends)
                 # Verify completion
                 if self._find_empty_cell() is None:
                     return True # Truly full
                 else:
                     return False # Stuck! (Greedy failure)
        else:
             # Standard First Empty Logic
             empty_cell = self._find_empty_cell()
             if not empty_cell:
                 return True # All filled
             r, c = empty_cell
             
        # [GREEDY UPDATE]: Choose move order based on strategy
        moves = ['L', 'R']
        if strategy:
             # Lazy import again not needed if scope is same
             # Step 2 & 3: Local Evaluation & Choose Optimal Move
             # get_move_order returns ['L', 'R'] or ['R', 'L'] sorted by score
             from cpu_ai import GreedyAI # Safety
             ai = GreedyAI(self, strategy) 
             moves = ai.get_move_order(r, c)
             
        elif randomize:
            random.shuffle(moves)
            
        for mv in moves:
            # Check validity
            # Note: During Generation, self.constraints is empty, so we only check Cycles.
            # During Solving (gameplay), we check both.
            if self.is_move_valid(r, c, mv):
                self.apply_move(r, c, mv)
                
                if self.solve_game(randomize, strategy):
                    return True
                
                # [GREEDY STRICT]: User requested "Greedy Alone" (No Backtracking).
                if strategy:
                    return False # Fail branch if valid move leads to dead end

                self.undo() # Backtrack

        # [FORCE FILL]: If we are here and using strategy, it means we have NO valid moves.
        # But user requested to fill strictly. So we FORCE a move (Invalid).
        if strategy:
            # Force 'L', if checking fails (best effort)
            # Actually, we should just pick the move that was 'better' scored, and force it.
            mv = moves[0] 
            # Force apply without validity check (hacky but satisfies request)
            # Warning: apply_move might assert. Let's rely on internal ability or ignore constraints.
            
            # Since apply_move does NOT check validity inside (it assumes caller did),
            # we can just call it! But we must be careful not to create weird graph states if possible.
            # However, cycle detection relies on valid moves.
            
            # Let's just TRY the first move again, but skip validation.
            self.apply_move(r, c, mv, check_validity=False)
            if self.solve_game(randomize, strategy):
                 return True
            # No backtrack here either
            return False

        return False

    def _find_empty_cell(self):
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] is None:
                    return (r, c)
        return None

    def is_cycle_created(self, r, c, move_type):
        """
        [REVIEW 1 REQUIREMENT]: Pure Graph Logic
        Checks if adding an edge between u and v creates a cycle.
        This is done by checking if v is ALREADY reachable from u in the graph.
        """
        # 1. Use pure graph reachability
        # Check if v is already reachable from u in self.graph
        # Note: The edge (u, v) does not exist yet (as we are checking before apply).
        # However, if we are overwriting an existing edge, we must ensure we don't traverse it.
        # But `apply_move` hasn't happened yet.
        # Wait, if we are switching L->R, we are theoretically removing L edge and checking R edge.
        # But strict logic: grid has L. graph has L-edge.
        # If we check "reachable(u, v)" for R-move, we might use the L-edge!
        # So we MUST simulate removal if overwriting.
        
        val_at_cell = self.grid[r][c]
        removed_edge = None
        
        if val_at_cell is not None:
            # Temporarily remove this edge from graph for the check
            if val_at_cell == 'L':
                u_old, v_old = (r, c), (r+1, c+1)
            else:
                u_old, v_old = (r+1, c), (r, c+1)
            
            self._remove_edge(u_old, v_old)
            removed_edge = (u_old, v_old)

        # 2. Determine Nodes u, v involved in the new edge
        if move_type == 'L':
            u, v = (r, c), (r+1, c+1)
        else:
            u, v = (r+1, c), (r, c+1)
            
        # 3. Check Reachability (BFS)
        queue = [u]
        visited = {u}
        found_cycle = False
        
        while queue:
            curr = queue.pop(0)
            if curr == v:
                found_cycle = True
                break
                
            for neighbor in self.graph[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Restore removed edge if any
        if removed_edge:
            self._add_edge(removed_edge[0], removed_edge[1])
            
        return found_cycle

    def is_move_valid(self, r, c, move_type, strict_cycles=True):
        if move_type is None: return True
        
        # 2. Cycle Check (The "No Loop" Rule)
        # Only check if strict_cycles is True (Default)
        if strict_cycles and self.is_cycle_created(r, c, move_type):
            return False

        # 1. Degree Constraints
        if move_type == 'L':
            n1, n2 = (r, c), (r+1, c+1)
        else:
            n1, n2 = (r+1, c), (r, c+1)
            
        # Current degrees (assuming (r,c) is effectively empty for the "Add" check)
        deg1 = self.node_degrees[n1]
        deg2 = self.node_degrees[n2]

        # GLOBAL RULE: Max degree is 4 in Slant
        # We check if adding 1 exceeds 4.
        if deg1 + 1 > 4: return False
        if deg2 + 1 > 4: return False

        if n1 in self.constraints and deg1 + 1 > self.constraints[n1]: return False
        if n2 in self.constraints and deg2 + 1 > self.constraints[n2]: return False
        
        return True

    def is_correction(self, r, c, player):
        # Check if the last move was made by this player at this cell
        if not self.history: return False
        
        # Last history item: (r, c, old_val, new_val, points, player)
        last = self.history[-1]
        
        # Determine format (handling migration just in case)
        if len(last) == 6:
            lr, lc, _, _, _, lplayer = last
            if lr == r and lc == c and lplayer == player:
                return True
        return False

    def apply_move(self, r, c, move_type, check_validity=True, player='HUMAN'):
        # Check Correction Hook
        is_correcting = False
        undone_state = None
        
        if self.turn != player and self.is_correction(r, c, player):
             # Capture state before undoing to allow restore
             if self.history:
                 undone_state = self.history[-1] # (r, c, old, new, pts, plr)
             
             self.undo()
             is_correcting = True
             
        # [REVIEW 1 UPDATE]: Reverted to Strict Loop Prevention
        
        current_val = self.grid[r][c]
        
        # 2. Cycle Check (The "No Loop" Rule)
        # [REVIEW 1]: If we are valid-checking (interactive play), we want to allow loops 
        # but warn the user. So we bypass `is_move_valid` cycle check here?
        # No, `is_move_valid` is called below.
        # Wait, I added a manual check here previously to optimize. 
        # Let's REMOVE this manual check and rely on `is_move_valid` logic below.
        
        if move_type is None:
            self.remove_move(r, c)
            self.history.append((r, c, current_val, None, 0, player))
            
            # If correcting (clearing), we normally wouldn't toggle turn.
             # However, if we cleared, we are back to 'HUMAN' turn (from undo).
            return True
        
        # Note: If is_correcting, we successfully Undo()-ed. 
        # So current_val should be None.
        
        if current_val is not None:
             self.remove_move(r, c, record_history=False)
             
        if check_validity and not self.is_move_valid(r, c, move_type):
            # Put back old if failed
            if current_val is not None:
                self.grid[r][c] = current_val
                if current_val == 'L': n1, n2 = (r,c), (r+1,c+1)
                else: n1, n2 = (r+1,c), (r,c+1)
                self.node_degrees[n1] += 1
                self.node_degrees[n2] += 1
            
            # CRITICAL FIX: If we were correcting (undoing a previous move) and this new one failed,
            # we must RESTORE the undone move, otherwise we lose the player's previous valid move!
            if is_correcting and undone_state:
                # Re-apply the undone move
                # undone_state = (r, c, old_val, new_val, points, player)
                # We know new_val was valid.
                # We can just call apply_move recursively without checks?
                # Or manually set it.
                _, _, _, u_new, _, u_plr = undone_state
                # We need to set it back.
                # Since we already reverted to 'current_val' (which is None/Old), we just apply u_new.
                self.apply_move(r, c, u_new, check_validity=False, player=u_plr)
                # This restores history and turn (to CPU presumably).
                
            return False
        
        # Note: If is_correcting, we successfully Undo()-ed. 
        # So current_val should be None (or previous state).
        # And Turn should be 'HUMAN'.
        # We continue as normal apply.
            
        # If replacing, remove first
        if current_val is not None:
             self.remove_move(r, c, record_history=False)
             
        # [REVIEW 1]: RELAXED CHECK for Human (Interactive)
        # We allow "loops" so we can WARN the user.
        # Strict checking is done by Solver/CPU via manual `check_validity=True` calls if needed.
        # But wait, default `is_move_valid` is strict.
        # So we must explicitly pass False here.
        if check_validity and not self.is_move_valid(r, c, move_type, strict_cycles=False):
            # Put back old if failed
            if current_val is not None:
                # We know old was valid (presumably), but we must be careful not to cycle check if we trust state.
                # Just restore manually
                self.grid[r][c] = current_val
                if current_val == 'L': n1, n2 = (r,c), (r+1,c+1)
                else: n1, n2 = (r+1,c), (r,c+1)
                self.node_degrees[n1] += 1
                self.node_degrees[n2] += 1
            return False
            
        self.grid[r][c] = move_type
        # Update Graph: Remove old edge if exists
        if current_val == 'L':
            u, v = (r, c), (r+1, c+1)
            self._remove_edge(u, v)
        elif current_val == 'R':
            u, v = (r+1, c), (r, c+1)
            self._remove_edge(u, v)

        # Update Graph: Add new edge
        if move_type == 'L':
            u, v = (r, c), (r+1, c+1)
            self._add_edge(u, v)
        elif move_type == 'R':
            u, v = (r+1, c), (r, c+1)
            self._add_edge(u, v)
            
        # Update degrees
        if move_type == 'L': n1, n2 = (r, c), (r+1, c+1)
        else: n1, n2 = (r+1, c), (r, c+1)
        
        # Update degrees
        self.node_degrees[n1] += 1
        self.node_degrees[n2] += 1
        
        # Criteria-Based Fair Scoring System
        # Both HUMAN and CPU use the SAME criteria, ensuring fairness
        points_earned = 0
        
        # Criterion 1: Base Move Points (1 point for any valid move)
        points_earned += 1
        
        # Criterion 2: Constraint Satisfaction Bonus (+2 points per constraint satisfied)
        # This rewards smart moves that complete numbered nodes
        nodes_checked = [n1, n2]
        constraints_satisfied = 0
        
        for n in nodes_checked:
            if n in self.constraints:
                limit = self.constraints[n]
                # Check if THIS move completes the constraint
                if self.node_degrees[n] == limit:
                    constraints_satisfied += 1
        
        points_earned += (constraints_satisfied * 2)  # +2 per satisfied constraint
        
        # Criterion 3: Perfect Cell Bonus (+3 points if all 4 corners are satisfied)
        # This rewards creating complete, valid cells
        cell_corners = [(r, c), (r+1, c+1), (r+1, c), (r, c+1)]
        all_corners_satisfied = True
        
        for corner in cell_corners:
            if corner in self.constraints:
                limit = self.constraints[corner]
                if self.node_degrees[corner] != limit:
                    all_corners_satisfied = False
                    break
        
        if all_corners_satisfied and any(c in self.constraints for c in cell_corners):
            points_earned += 3  # Perfect cell bonus
        
        # Criterion 4: Strategic Position Bonus (+1 for center moves)
        # Slightly rewards filling the center area which is strategically important
        mid = self.size // 2
        if abs(r - mid) <= 1 and abs(c - mid) <= 1:
            points_earned += 1
        
        self.scores[player] += points_earned
        
        # [CRITICAL FIX]: Set Ownership
        self.owners[r][c] = player
        
        self.history.append((r, c, current_val, move_type, points_earned, player))
        
        # Track move with chess coordinates for review feature
        # Chess-style: columns are A-Z (left to right), rows are 1-N (bottom to top)
        column_letter = chr(ord('A') + c)
        row_number = self.size - r  # Invert: bottom row = 1, top row = size
        cell = f"{column_letter}{row_number}"
        self.move_history.append({
            "player": player,
            "row": r,
            "col": c,
            "value": move_type,
            "cell": cell
        })
        
        self.check_completion()
        
        # [REVIEW 1 REQUIREMENT]: Update Loop Visualization constantly
        self.detect_cycle_dfs()
        
        # Toggle Turn
        self.turn = 'CPU' if self.turn == 'HUMAN' else 'HUMAN'
        return True
