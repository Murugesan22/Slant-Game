"""
Algorithmic AI for Slant Game

Strictly uses algorithmic approaches (DP, D&C, Hybrid, Cut) to solve puzzles.
NO greedy heuristics, NO random selection, NO evaluation functions.

The AI precomputes a complete solution using the selected algorithm,
then follows that solution move-by-move during gameplay.
"""

import copy


class AlgorithmicAI:
    """
    AI that uses pure algorithmic solvers (DP, D&C, Hybrid, Cut).
    
    Workflow:
    1. On initialization, precompute full solution using selected solver
    2. During gameplay, return next move from precomputed solution
    3. If user deviates, recompute solution and continue
    
    NO greedy logic, NO heuristics, NO scoring - strictly algorithmic.
    """
    
    def __init__(self, game, solver_type='dp'):
        """
        Initialize the Algorithmic AI.
        
        Args:
            game: SlantGame instance
            solver_type: 'dp', 'dnc', 'hybrid', or 'cut'
        """
        self.game = game
        self.solver_type = solver_type
        self.solution_grid = None
        self.solution_computed = False
        
    def compute_solution(self):
        """
        Precompute full solution using selected solver.
        
        Returns:
            bool: True if solution found, False otherwise
        """
        if self.solver_type == 'dp':
            from dp_solver import solve_and_extract
            success, solution = solve_and_extract(self.game, use_enhanced=True)
        elif self.solver_type == 'dnc':
            from dnc_solver import solve_and_extract_dnc
            success, solution = solve_and_extract_dnc(self.game)
        elif self.solver_type == 'hybrid':
            from dnc_solver import solve_and_extract_hybrid
            success, solution = solve_and_extract_hybrid(self.game)
        elif self.solver_type == 'cut':
            from dnc_solver import solve_and_extract_cut
            success, solution = solve_and_extract_cut(self.game)
        else:
            # Default to DP if unknown solver type
            from dp_solver import solve_and_extract
            success, solution = solve_and_extract(self.game, use_enhanced=True)
        
        if success:
            self.solution_grid = solution
            self.solution_computed = True
            return True
        else:
            self.solution_grid = None
            self.solution_computed = False
            return False
    
    def get_next_move(self):
        """
        Get next move from precomputed solution.
        
        Finds the first empty cell and returns the solution move for it.
        NO evaluation, NO scoring, NO greedy decision.
        
        Returns:
            tuple: (r, c, move_type) or None if no move available
        """
        # If solution not computed yet, compute it
        if not self.solution_computed:
            if not self.compute_solution():
                return None  # No solution exists
        
        # Find first empty cell
        for r in range(self.game.size):
            for c in range(self.game.size):
                if self.game.grid[r][c] is None:
                    # Return solution move for this cell
                    if self.solution_grid and r < len(self.solution_grid) and c < len(self.solution_grid[r]):
                        move_type = self.solution_grid[r][c]
                        if move_type in ['L', 'R']:
                            return (r, c, move_type)
        
        # No empty cells found
        return None
    
    def recompute_if_deviated(self):
        """
        Check if current game state matches precomputed solution.
        If user deviated, recompute solution.
        
        Returns:
            bool: True if recomputation successful or not needed
        """
        if not self.solution_computed:
            return self.compute_solution()
        
        # Check if current state matches solution
        deviated = False
        for r in range(self.game.size):
            for c in range(self.game.size):
                current = self.game.grid[r][c]
                expected = self.solution_grid[r][c] if self.solution_grid else None
                
                if current is not None and current != expected:
                    deviated = True
                    break
            if deviated:
                break
        
        if deviated:
            # Recompute solution from current state
            return self.compute_solution()
        
        return True


def get_algorithmic_move(game, solver_type='dp'):
    """
    Standalone function to get a move using algorithmic approach.
    
    Args:
        game: SlantGame instance
        solver_type: 'dp', 'dnc', 'hybrid', or 'cut'
    
    Returns:
        tuple: (r, c, move_type) or None if no move available
    """
    ai = AlgorithmicAI(game, solver_type)
    return ai.get_next_move()
