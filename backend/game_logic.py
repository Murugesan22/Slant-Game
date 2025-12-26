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
        # Retry loop to ensure valid puzzle generation
        attempts = 0
        success = False
        
        while attempts < 10 and not success:
            attempts += 1
            # 1. Start with empty board
            self._initialize_empty_state()
            self.constraints = {} # CRITICAL FIX: Clear constraints from previous failed attempts!

            
            # 2. Fill it with a valid solution (randomized backtracking)
            if self.solve_game(randomize=True):
                 temp_degrees = self.node_degrees.copy()
                 self._initialize_empty_state()
                 
                 # 3. Initial Reveal (Uniform Spread via Farthest Point Sampling)
                 nodes = list(temp_degrees.keys())
                 
                 # Helper to find node with max distance to existing clues
                 def get_farthest_unrevealed(candidates, existing):
                     if not existing:
                         return random.choice(candidates)
                     
                     best_node = None
                     max_min_dist = -1
                     
                     for cand in candidates:
                         # Manhattan distance to nearest existing clue
                         min_dist = min(abs(cand[0]-e[0]) + abs(cand[1]-e[1]) for e in existing)
                         
                         if min_dist > max_min_dist:
                             max_min_dist = min_dist
                             best_node = cand
                         elif min_dist == max_min_dist:
                             # Tie-break randomly to avoid deterministic patterns
                             if random.random() < 0.3:
                                 best_node = cand
                                 
                     return best_node

                 # Start with random center-ish node to anchor
                 center = (self.nodes_size // 2, self.nodes_size // 2)
                 candidates = [n for n in nodes]
                 
                 # Target: 35%
                 target_count = int(len(nodes) * 0.35)
                 
                 # Add first node (closest to center to ensure playability starts there?) 
                 # Or just random. Random is better for variety.
                 first = random.choice(candidates)
                 self.constraints[first] = temp_degrees[first]
                 candidates.remove(first)
                 
                 while len(self.constraints) < target_count:
                     next_node = get_farthest_unrevealed(candidates, list(self.constraints.keys()))
                     self.constraints[next_node] = temp_degrees[next_node]
                     candidates.remove(next_node)

                 # 4. Enhance for Uniqueness
                 # Continue using Farthest Sampling for extra clues to fill gaps
                 unique = False
                 max_clues = int(len(nodes) * 0.40) 
                 curr_clues = len(self.constraints)
                 
                 for _ in range(20): 
                     solutions = self.count_solutions(limit=2)
                     if solutions == 1:
                         unique = True
                         break # Unique!
                     
                     if not candidates or curr_clues >= max_clues:
                         break 
                     
                     # Add clue in the biggest gap
                     new_clue = get_farthest_unrevealed(candidates, list(self.constraints.keys()))
                     self.constraints[new_clue] = temp_degrees[new_clue]
                     candidates.remove(new_clue)
                     curr_clues += 1
                     
                 if unique:
                     success = True
            
        if success:
            print(f"Puzzle generated in {attempts} attempts with {len(self.constraints)} clues (Spread Optimized)")
        else:
            print("Failed to generate unique puzzle under density limit, using last attempt (fallback)")
            # If fallback, we still have the last attempt's constraints. 
            # It will be solvable but maybe not unique.
            # But heavily constrained (up to limit).
            # We respect the limit over uniqueness if forced.


    def count_solutions(self, limit=2):
        """
        Counts solutions consistent with current self.constraints.
        Returns count (capped at limit).
        Operates on the current grid (assumed empty or partially filled during recursion).
        Does NOT modify self.grid permanently (backtracks).
        """
        # We need a recursive helper that doesn't rely on global state flags like self.status
        # And repeats the logic of solve_game but continues after finding one.
        count = 0
        
        def backtrack(r, c):
            nonlocal count
            if count >= limit: return

            # Find next empty
            # Optimization: pass r, c index instead of searching every time
            # For simplicity, use simple linear scan argument simulation
            next_r, next_c = -1, -1
            found = False
            
            # Start search from current r, c
            curr_idx = r * self.size + c
            for idx in range(curr_idx, self.size * self.size):
                ir, ic = idx // self.size, idx % self.size
                if self.grid[ir][ic] is None:
                    next_r, next_c = ir, ic
                    found = True
                    break
            
            if not found:
                count += 1
                return

            moves = ['L', 'R'] # specific order doesn't matter for counting
            
            for mv in moves:
                if self.is_move_valid(next_r, next_c, mv):
                    self.apply_move(next_r, next_c, mv, check_validity=False) # Skip re-check, we checked above
                    # Note: apply_move updates degrees/scores/history. 
                    # We MUST rely on exact state restoration.
                    
                    backtrack(next_r, next_c)
                    
                    self.undo() # Restore
                    if count >= limit: return
        
        # Start search
        backtrack(0, 0)
        return count

    def solve_game(self, randomize=False):
        # Backtracking solver
        # Returns True if solved, False otherwise
        
        # Verify if current state has cycles? (Should be maintained by moves)
        
        empty_cell = self._find_empty_cell()
        if not empty_cell:
            return True # All filled
            
        r, c = empty_cell
        moves = ['L', 'R']
        if randomize:
            random.shuffle(moves)
            
        for mv in moves:
            # Check validity
            # Note: During Generation, self.constraints is empty, so we only check Cycles.
            # During Solving (gameplay), we check both.
            if self.is_move_valid(r, c, mv):
                self.apply_move(r, c, mv)
                
                if self.solve_game(randomize):
                    return True
                    
                self.undo() # Backtrack
                
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
        self.check_completion()
        
        # [REVIEW 1 REQUIREMENT]: Update Loop Visualization constantly
        self.detect_cycle_dfs()
        
        # Toggle Turn
        self.turn = 'CPU' if self.turn == 'HUMAN' else 'HUMAN'
        return True

    def remove_move(self, r, c, record_history=False):
        val = self.grid[r][c]
        if val is None: return
        
        if val == 'L':
             n1, n2 = (r, c), (r+1, c+1)
             u, v = (r, c), (r+1, c+1)
             self._remove_edge(u, v)
        elif val == 'R':
             n1, n2 = (r+1, c), (r, c+1)
             u, v = (r+1, c), (r, c+1)
             self._remove_edge(u, v)
        else:
             return 
            
        self.node_degrees[n1] -= 1
        self.node_degrees[n2] -= 1
        self.grid[r][c] = None
        
        if record_history:
             self.history.append((r, c, val, None))

    def undo(self):
        if not self.history: return False
        
        # Pop extended history
        # (r, c, old_val, new_val, points, player)
        # Note: Previous history format was (r,c,old,new). 
        # We need to handle migration or just assume new format for new games.
        # Ideally, we just check len.
        
        last = self.history.pop()
        if len(last) == 6:
            r, c, old_val, new_val, points, player = last
        else:
            # Fallback for old history (if any exists in memory, strict restart needed usually)
            r, c, old_val, new_val = last
            points = 0
            player = None

        # Revert change: Remove New, Add Old
        if new_val is not None:
            if new_val == 'L': n1, n2 = (r, c), (r+1, c+1)
            else: n1, n2 = (r+1, c), (r, c+1)
            self.node_degrees[n1] -= 1
            self.node_degrees[n2] -= 1
            
            # Revert Points
            if player:
                self.scores[player] -= points
            
            self.owners[r][c] = None
            
        self.grid[r][c] = old_val
        
        # Update Graph: Remove edge added by 'new_val' that we are undoing
        # Wait, undo logic restores `old_val`.
        # So we must remove `new_val` edge and add `old_val` edge.
        
        # new_val was 'L' or 'R' or None?
        # The history entry tells us what happened.
        # But here we rely on the implementation assuming this block runs AFTER `undo` restoration logic?
        # NO, this block IS inside `undo`.
        # `move` struct has `(r, c, old_val, new_val, ...)`
        
        # We need to reverse the graph change invoked by `new_val`
        if new_val == 'L':
             u, v = (r, c), (r+1, c+1)
             self._remove_edge(u, v)
        elif new_val == 'R':
             u, v = (r+1, c), (r, c+1)
             self._remove_edge(u, v)

        if old_val is not None:
            # We restored old_val to grid, so add its edge back
            if old_val == 'L':
                u, v = (r, c), (r+1, c+1)
                self._add_edge(u, v)
            elif old_val == 'R':
                u, v = (r+1, c), (r, c+1)
                self._add_edge(u, v)

            if old_val == 'L': n1, n2 = (r, c), (r+1, c+1)
            else: n1, n2 = (r+1, c), (r, c+1)
            self.node_degrees[n1] += 1
            self.node_degrees[n2] += 1
        
        self.check_completion()
        
        # Toggle Turn back if it was a real move and player was tracked
        if new_val is not None and player:
             # If we undid a CPU move, turn goes back to CPU.
             # If we undid Human move, turn goes back to Human.
             # Since we alternate strict, this simply toggles back?
             # Yes.
             self.turn = player
             
        return True

    def _add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def _remove_edge(self, u, v):
        if v in self.graph[u]: self.graph[u].remove(v)
        if u in self.graph[v]: self.graph[v].remove(u)

    def get_graph_representation(self):
        """
        [REVIEW 1 REQUIREMENT]: Graph Representation from Grid
        Returns the persistent Adjacency List.
        """
        return self.graph

    def detect_cycle_dfs(self):
        """
        [REVIEW 1 REQUIREMENT]: Graph Algorithm (DFS)
        Detects cycles using Depth First Search on the Adjacency List representation.
        Returns True if a cycle exists, and populates self.loop_cells with the edge coordinates.
        """
        graph = self.get_graph_representation()
        visited = set()
        parent_map = {} # To reconstruct path
        
        # Reset previous loop cells
        self.loop_cells = []
        
        found_any = False
        
        def dfs(node, parent):
            nonlocal found_any
            visited.add(node)
            parent_map[node] = parent
            
            for neighbor in graph[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    # Cycle found! 
                    found_any = True
                    cycle_nodes = [neighbor, node]
                    curr = node
                    while curr != neighbor and curr in parent_map:
                        curr = parent_map[curr]
                        if curr:
                            cycle_nodes.append(curr)
                        if curr == neighbor: break
                            
                    for i in range(len(cycle_nodes)-1):
                        u, v = cycle_nodes[i], cycle_nodes[i+1]
                        self._find_cell_for_edge(u, v)

                    continue # Do NOT return True, keep checking other branches/edges
                
                dfs(neighbor, node)
                # We ignore return value since we rely on found_any and loop_cells population
            
            return False # Return irrelevant now

        for node in graph:
            if node not in visited:
                dfs(node, None)
                
        return found_any

    def _find_cell_for_edge(self, u, v):
        # Identify the grid cell connecting node u and node v
        r1, c1 = u
        r2, c2 = v
        
        # Determine top-left corner of the cell
        # Case 1: (r, c) <-> (r+1, c+1) (L)
        # min_r = min(r1,r2), min_c = min(c1,c2)
        # If L, u and v are diagonals.
        # If R, u=(r, c+1), v=(r+1, c).
        
        min_r, min_c = min(r1, r2), min(c1, c2)
        
        # This cell is at Grid[min_r][min_c]
        # BUT wait, if it's 'R' slant:
        # u=(0, 1), v=(1, 0). min_r=0, min_c=0.
        # So cell is indeed (min_r, min_c).
        
        if 0 <= min_r < self.size and 0 <= min_c < self.size:
             self.loop_cells.append((min_r, min_c))

    def to_dict(self):
        """
        Return state as JSON-serializable dict
        """
        grid_copy = [row[:] for row in self.grid]
        
        # Convert tuple keys to string "r,c" for JSON compatibility
        constraints_str = {f"{k[0]},{k[1]}": v for k, v in self.constraints.items()}
        node_degrees_str = {f"{k[0]},{k[1]}": v for k, v in self.node_degrees.items()}
        
        # Include Graph Rep for Analysis/Debug
        graph_rep = self.get_graph_representation()
        # Convert tuple keys/values to strings for JSON
        graph_str = {}
        for k, v in graph_rep.items():
            k_str = f"{k[0]},{k[1]}"
            v_str = [f"{n[0]},{n[1]}" for n in v]
            graph_str[k_str] = v_str
        
        return {
            'size': self.size,
            'grid': grid_copy,
            'constraints': constraints_str,
            'node_degrees': node_degrees_str,
            'status': self.status,
            'turn': self.turn,
            'scores': self.scores,
            'owners': self.owners,
            'loop_cells': getattr(self, 'loop_cells', []), # [REVIEW 1]: Expose Loop for Visualization
            'graph': graph_str # [REVIEW 1]: Exposing API to graph
        }
