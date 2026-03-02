"""
Algorithmic AI for Slant Game

Strictly uses algorithmic approaches (DP, D&C, Hybrid, Cut) to solve puzzles.
NO greedy heuristics, NO random selection, NO evaluation functions.

The AI precomputes a complete solution using the selected algorithm,
then follows that solution move-by-move during gameplay.
"""

import copy
import threading


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
        
    def compute_solution(self, timeout=10):
        """
        Precompute full solution using selected solver with timeout.
        
        Runs the solver in a daemon thread. If it doesn't finish within
        `timeout` seconds, returns False (CPU passes turn).
        
        Args:
            timeout: Maximum seconds to allow the solver to run (default: 10)
        
        Returns:
            bool: True if solution found, False otherwise (including timeout)
        """
        result = {'success': False, 'solution': None}
        
        def _run_solver():
            try:
                if self.solver_type == 'dp':
                    from dp_solver import solve_and_extract
                    s, sol = solve_and_extract(self.game, use_enhanced=True)
                elif self.solver_type == 'dnc':
                    from dnc_solver import solve_and_extract_dnc
                    s, sol = solve_and_extract_dnc(self.game)
                elif self.solver_type == 'hybrid':
                    from dnc_solver import solve_and_extract_hybrid
                    s, sol = solve_and_extract_hybrid(self.game)
                elif self.solver_type == 'cut':
                    from dnc_solver import solve_and_extract_cut
                    s, sol = solve_and_extract_cut(self.game)
                elif self.solver_type == 'mrv':
                    from mrv_solver import solve_and_extract
                    s, sol = solve_and_extract(self.game)
                elif self.solver_type == 'cbj':
                    from cbj_solver import solve_and_extract
                    s, sol = solve_and_extract(self.game)
                elif self.solver_type == 'fc':
                    from fc_solver import solve_and_extract
                    s, sol = solve_and_extract(self.game)
                else:
                    from dp_solver import solve_and_extract
                    s, sol = solve_and_extract(self.game, use_enhanced=True)
                result['success'] = s
                result['solution'] = sol
            except Exception as e:
                print(f"[CPU_AI] Solver error: {e}")
                result['success'] = False
                result['solution'] = None
        
        solver_thread = threading.Thread(target=_run_solver, daemon=True)
        solver_thread.start()
        solver_thread.join(timeout=timeout)
        
        if solver_thread.is_alive():
            # Solver timed out — it will keep running in daemon thread
            # but we return False immediately
            print(f"[CPU_AI] Solver timed out after {timeout}s")
            self.solution_grid = None
            self.solution_computed = False
            return False
        
        if result['success']:
            self.solution_grid = result['solution']
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
        If no valid solution exists, tries to find a valid move using
        is_move_valid before falling back to a forced move.
        
        Returns:
            tuple: (r, c, move_type) or None if board is already full
        """
        # Try to compute solution if not done yet
        if not self.solution_computed:
            self.compute_solution()  # May fail — that's ok, we'll force moves
        
        # Find first empty cell
        for r in range(self.game.size):
            for c in range(self.game.size):
                if self.game.grid[r][c] is None:
                    # If we have a valid solution, use it
                    if self.solution_grid and r < len(self.solution_grid) and c < len(self.solution_grid[r]):
                        move_type = self.solution_grid[r][c]
                        if move_type in ['L', 'R']:
                            return (r, c, move_type)
                    # No solution — try to find a valid move first
                    for mv in ['L', 'R']:
                        if self.game.is_move_valid(r, c, mv):
                            return (r, c, mv)
                    # Neither is valid — force 'L' to fill the board
                    return (r, c, 'L')
        
        # No empty cells found — board is full
        return None
    
    def recompute_if_deviated(self):
        """
        Check if current game state matches precomputed solution.
        If user deviated, recompute solution.
        
        Always returns True — CPU will force moves even without a solution.
        """
        if not self.solution_computed:
            self.compute_solution()  # Try, but don't fail if it can't solve
            return True
        
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
            # Recompute solution from current state (may fail — that's ok)
            self.compute_solution()
        
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
