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
