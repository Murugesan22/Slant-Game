import random


def quicksort(arr, key=None, reverse=False):
    """
    Custom Quicksort implementation with key function support.
    
    Educational implementation demonstrating divide-and-conquer algorithm.
    Uses Lomuto partition scheme with randomized pivot selection.
    
    Args:
        arr: List to sort (modified in-place)
        key: Function to extract comparison key from each element
        reverse: Sort in descending order if True
    
    Time Complexity: O(n log n) average case, O(n²) worst case
    Space Complexity: O(log n) for recursion stack
    """
    def _partition(arr, low, high, key_func, reverse):
        """Lomuto partition scheme with randomized pivot"""
        # Randomize pivot to avoid worst case on sorted data
        pivot_idx = random.randint(low, high)
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
        
        pivot_val = key_func(arr[high]) if key_func else arr[high]
        i = low - 1
        
        for j in range(low, high):
            curr_val = key_func(arr[j]) if key_func else arr[j]
            
            # Comparison based on reverse flag
            if (curr_val <= pivot_val) if not reverse else (curr_val >= pivot_val):
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    def _quicksort_recursive(arr, low, high, key_func, reverse):
        """Recursive quicksort helper"""
        if low < high:
            pi = _partition(arr, low, high, key_func, reverse)
            _quicksort_recursive(arr, low, pi - 1, key_func, reverse)
            _quicksort_recursive(arr, pi + 1, high, key_func, reverse)
    
    if len(arr) <= 1:
        return
    
    _quicksort_recursive(arr, 0, len(arr) - 1, key, reverse)

class GreedyAI:
    def __init__(self, game, strategy=1):
        self.game = game
        self.strategy = strategy  # 1, 2, or 3
        
    def get_best_move(self):
        """Returns (r, c, move_type) based on selected strategy"""
        if self.strategy == 1:
            return self._strategy_constraint_focused()
        elif self.strategy == 2:
            return self._strategy_edge_first()
        elif self.strategy == 3:
            return self._strategy_random_greedy()
        else:
            return self._strategy_constraint_focused()  # Default
    
    # ==================== STRATEGY 1: Constraint-Focused ====================
    def _strategy_constraint_focused(self):
        """Original strategy: Prioritizes cells near constraints"""
        best_score = -float('inf')
        best_moves = []
        
        size = self.game.size
        
        # Priority Sort: Process cells with nearby constraints first
        cells = []
        for r in range(size):
            for c in range(size):
                if self.game.grid[r][c] is None:
                    # Count adjacent constraints
                    constraints_nearby = 0
                    nodes = [(r, c), (r+1, c+1), (r+1, c), (r, c+1)]
                    for n in nodes:
                        if n in self.game.constraints:
                            constraints_nearby += 1
                    cells.append(((r, c), constraints_nearby))
                    
        # Sort desc by constraints count using custom Quicksort
        quicksort(cells, key=lambda x: x[1], reverse=True)
        
        for (r, c), _ in cells:
            for move_type in ['L', 'R']:
                if self.game.is_move_valid(r, c, move_type):
                    score = self._evaluate_constraint_focused(r, c, move_type)
                    if score > best_score:
                        best_score = score
                        best_moves = [(r, c, move_type)]
                    elif score == best_score:
                        best_moves.append((r, c, move_type))
        
        if not best_moves:
            return None
            
        return random.choice(best_moves)
    
    def _evaluate_constraint_focused(self, r, c, move_type):
        """Evaluation for constraint-focused strategy"""
        score = 1.0
        
        if move_type == 'L':
            nodes = [(r, c), (r+1, c+1)]
        else:
            nodes = [(r+1, c), (r, c+1)]
            
        for node in nodes:
            current_deg = self.game.node_degrees[node]
            limit = self.game.constraints.get(node)
            
            if limit is not None:
                new_deg = current_deg + 1
                if new_deg == limit:
                    score += 0.5  # Bonus for satisfying constraint
                elif new_deg < limit:
                    score += 0.2  # Prefer moves toward constraints
                else:
                    score -= 100  # Invalid
            else:
                if current_deg + 1 > 4:
                    score -= 100
                    
        # Bonus for connecting existing lines
        for node in nodes:
            current_deg = self.game.node_degrees[node]
            if current_deg > 0:
                score += 0.1
        
        # Center preference
        mid = self.game.size // 2
        dist = abs(r - mid) + abs(c - mid)
        score -= (dist * 0.05)
        
        return score
    
    # ==================== STRATEGY 2: Edge-First ====================
    def _strategy_edge_first(self):
        """Strategy 2: Starts from edges and works inward"""
        best_score = -float('inf')
        best_moves = []
        
        size = self.game.size
        
        # Priority: edges and corners first (distance from center)
        cells = []
        for r in range(size):
            for c in range(size):
                if self.game.grid[r][c] is None:
                    # Calculate distance from center (higher = edge)
                    mid = size / 2
                    edge_priority = abs(r - mid) + abs(c - mid)
                    cells.append(((r, c), edge_priority))
        
        # Sort by edge priority (descending - edges first) using custom Quicksort
        quicksort(cells, key=lambda x: x[1], reverse=True)
        
        for (r, c), _ in cells:
            for move_type in ['L', 'R']:
                if self.game.is_move_valid(r, c, move_type):
                    score = self._evaluate_edge_first(r, c, move_type)
                    if score > best_score:
                        best_score = score
                        best_moves = [(r, c, move_type)]
                    elif score == best_score:
                        best_moves.append((r, c, move_type))
        
        if not best_moves:
            return None
            
        return random.choice(best_moves)
    
    def _evaluate_edge_first(self, r, c, move_type):
        """Evaluation for edge-first strategy"""
        score = 1.0
        
        if move_type == 'L':
            nodes = [(r, c), (r+1, c+1)]
        else:
            nodes = [(r+1, c), (r, c+1)]
        
        # Check validity
        for node in nodes:
            current_deg = self.game.node_degrees[node]
            limit = self.game.constraints.get(node)
            
            if limit is not None:
                new_deg = current_deg + 1
                if new_deg == limit:
                    score += 0.6
                elif new_deg < limit:
                    score += 0.3
                else:
                    score -= 100
            else:
                if current_deg + 1 > 4:
                    score -= 100
        
        # EDGE PREFERENCE: Higher score for cells farther from center
        mid = self.game.size / 2
        edge_dist = abs(r - mid) + abs(c - mid)
        score += (edge_dist * 0.15)  # Bonus for being near edge
        
        # Small bonus for line continuity
        for node in nodes:
            if self.game.node_degrees[node] > 0:
                score += 0.05
        
        return score
    
    # ==================== STRATEGY 3: Random-Greedy ====================
    def _strategy_random_greedy(self):
        """Strategy 3: Random selection among valid moves with basic scoring"""
        valid_moves = []
        
        size = self.game.size
        
        # Collect all valid moves
        for r in range(size):
            for c in range(size):
                if self.game.grid[r][c] is None:
                    for move_type in ['L', 'R']:
                        if self.game.is_move_valid(r, c, move_type):
                            score = self._evaluate_random_greedy(r, c, move_type)
                            # Only consider moves with positive scores
                            if score > 0:
                                valid_moves.append((r, c, move_type, score))
        
        if not valid_moves:
            return None
        
        # Sort by score but with intentional randomness
        # Add random noise to scores to make it less predictable
        noisy_moves = []
        for r, c, move_type, score in valid_moves:
            noise = random.uniform(-0.3, 0.3)  # Random variation
            noisy_moves.append((r, c, move_type, score + noise))
        
        # Sort and pick from top candidates using custom Quicksort
        quicksort(noisy_moves, key=lambda x: x[3], reverse=True)
        
        # Pick from top 30% of moves to add variety
        top_count = max(1, len(noisy_moves) // 3)
        chosen = random.choice(noisy_moves[:top_count])
        
        return (chosen[0], chosen[1], chosen[2])
    
    def _evaluate_random_greedy(self, r, c, move_type):
        """Simpler evaluation for random-greedy strategy"""
        score = 1.0
        
        if move_type == 'L':
            nodes = [(r, c), (r+1, c+1)]
        else:
            nodes = [(r+1, c), (r, c+1)]
        
        # Basic validity check
        for node in nodes:
            current_deg = self.game.node_degrees[node]
            limit = self.game.constraints.get(node)
            
            if limit is not None:
                new_deg = current_deg + 1
                if new_deg == limit:
                    score += 0.4
                elif new_deg < limit:
                    score += 0.1
                else:
                    score -= 100
            else:
                if current_deg + 1 > 4:
                    score -= 100
        
        # Light preference for connecting lines
        for node in nodes:
            if self.game.node_degrees[node] > 0:
                score += 0.08
        
        return score
    # ==================== HELPERS FOR SOLVER GUIDANCE ====================
    def get_best_empty_cell(self):
        """
        Returns the (r, c) of the empty cell with highest priority according to strategy.
        Used by the backtracking solver to pick 'which cell to fill next'.
        """
        size = self.game.size
        candidates = []
        
        # 1. Collect all empty cells
        for r in range(size):
            for c in range(size):
                if self.game.grid[r][c] is None:
                    # Score the CELL itself (not the move)
                    priority = 0
                    
                    # [SMART GREEDY]: Prioritize FORCED MOVES (Unit Propagation)
                    # If a cell has only 1 valid move, we MUST take it now to avoid conflicts.
                    valid_opts = 0
                    if self.game.is_move_valid(r, c, 'L'): valid_opts += 1
                    if self.game.is_move_valid(r, c, 'R'): valid_opts += 1
                    
                    if valid_opts == 1:
                        priority += 1000000 # Forced move! Immediate priority.
                    elif valid_opts == 0:
                        priority = -1000000 # Dead end (cannot fill).
                    
                    if self.strategy == 1:
                        # CELL PRIORITY: Number of adjacent constraints
                        # (Same logic as _strategy_constraint_focused sort)
                        nodes = [(r, c), (r+1, c+1), (r+1, c), (r, c+1)]
                        for n in nodes:
                            if n in self.game.constraints:
                                priority += 10 # High weight for constraints
                        # Tie-breaker: Distance from center (closer is better)
                        mid = size // 2
                        dist = abs(r - mid) + abs(c - mid)
                        priority += (10 - dist) * 0.1 
                        
                    elif self.strategy == 2:
                        # CELL PRIORITY: Edge distance (Farther is better)
                        mid = size / 2
                        dist = abs(r - mid) + abs(c - mid)
                        priority += dist * 10
                        
                    elif self.strategy == 3:
                        # CELL PRIORITY: Random
                        priority += random.random()
                        
                    candidates.append(((r, c), priority))
        
        if not candidates:
            return None
            
        # 2. Sort safely
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # [FORCE FILL UPDATE]: Do NOT return None if dead end.
        # We want to return the cell so the solver can Force-Fill it.
        # if candidates[0][1] < 0:
        #    return None

        return candidates[0][0]

    def get_move_order(self, r, c):
        """
        Returns ['L', 'R'] or ['R', 'L'] sorted by heuristic score for this cell.
        Includes 1-Step Look-ahead to avoid immediate dead ends.
        """
        # Score both
        base_L = 0
        base_R = 0
        
        if self.strategy == 1:
            base_L = self._evaluate_constraint_focused(r, c, 'L')
            base_R = self._evaluate_constraint_focused(r, c, 'R')
        elif self.strategy == 2:
            base_L = self._evaluate_edge_first(r, c, 'L')
            base_R = self._evaluate_edge_first(r, c, 'R')
        elif self.strategy == 3:
            base_L = self._evaluate_random_greedy(r, c, 'L') + random.uniform(-0.1, 0.1)
            base_R = self._evaluate_random_greedy(r, c, 'R') + random.uniform(-0.1, 0.1)
        else:
            base_L = self._evaluate_constraint_focused(r, c, 'L')
            base_R = self._evaluate_constraint_focused(r, c, 'R')

        # [BASIC GREEDY]: Removed advanced Look-Ahead.
        # Decisions are now based purely on standard heuristics (Constraint/Edge).
        
        final_L = base_L
        final_R = base_R
            
        if final_L >= final_R:
            return ['L', 'R']
        else:
            return ['R', 'L']

    def _causes_dead_end(self, r, c):
        """
        Check if any neighbor of (r, c) becomes unfillable (0 valid moves)
        """
        size = self.game.size
        # Neighbors
        neighbors = []
        if r > 0: neighbors.append((r-1, c))
        if r < size-1: neighbors.append((r+1, c))
        if c > 0: neighbors.append((r, c-1))
        if c < size-1: neighbors.append((r, c+1))
        
        for nr, nc in neighbors:
            if self.game.grid[nr][nc] is None:
                valid = 0
                if self.game.is_move_valid(nr, nc, 'L'): valid += 1
                if self.game.is_move_valid(nr, nc, 'R'): valid += 1
                
                if valid == 0:
                    return True # Detected a dead end!
        return False
