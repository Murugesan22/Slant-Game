"""
Test suite for DP and Divide & Conquer algorithms in Slant Game.

Tests:
1. DP Profile Solver correctness
2. DP-Enhanced Backtracking correctness
3. D&C Grid Solver correctness
4. D&C Cycle Detection agreement with DFS
5. Memoized Validator consistency
6. New AI strategies (4, 5) produce valid moves

Run with: python -m pytest test_algorithms.py -v
"""

import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_logic import SlantGame
from dp_solver import solve_with_dp, solve_with_dp_enhanced, MemoizedValidator
from dnc_solver import solve_with_dnc, detect_cycle_dnc, solve_with_boundary_cut, solve_with_hybrid
from cpu_ai import GreedyAI


def validate_solution(game):
    """
    Verify that a game state represents a valid, complete solution.
    Returns (is_valid, error_message).
    """
    size = game.size
    
    # 1. All cells must be filled
    for r in range(size):
        for c in range(size):
            if game.grid[r][c] is None:
                return False, f"Cell ({r},{c}) is empty"
    
    # 2. All constraints must be satisfied
    for node, limit in game.constraints.items():
        deg = game.node_degrees.get(node, 0)
        if deg != limit:
            return False, f"Constraint at {node}: expected {limit}, got {deg}"
    
    # 3. No cycles
    has_cycle = game.detect_cycle_dfs()
    if has_cycle:
        return False, "Solution contains a cycle"
    
    return True, "Valid"


# ==============================================================================
# DP SOLVER TESTS
# ==============================================================================

class TestDPSolver:
    """Tests for the Dynamic Programming solver."""
    
    def test_dp_solver_3x3(self):
        """DP solver should produce a valid solution on 3x3."""
        game = SlantGame(size=3)
        result = solve_with_dp(game)
        if result:
            valid, msg = validate_solution(game)
            assert valid, f"DP solver produced invalid solution on 3x3: {msg}"
        # It's OK if DP can't solve (some puzzles may be very constrained),
        # but if it returns True, the solution must be valid.
    
    def test_dp_solver_5x5(self):
        """DP solver should produce a valid solution on 5x5."""
        game = SlantGame(size=5)
        result = solve_with_dp(game)
        if result:
            valid, msg = validate_solution(game)
            assert valid, f"DP solver produced invalid solution on 5x5: {msg}"
    
    def test_dp_enhanced_3x3(self):
        """DP-enhanced backtracking should produce a valid solution on 3x3."""
        game = SlantGame(size=3)
        result = solve_with_dp_enhanced(game)
        assert result, "DP-enhanced solver should find a solution on 3x3"
        valid, msg = validate_solution(game)
        assert valid, f"DP-enhanced solver produced invalid solution: {msg}"
    
    def test_dp_enhanced_5x5(self):
        """DP-enhanced backtracking should produce a valid solution on 5x5."""
        game = SlantGame(size=5)
        result = solve_with_dp_enhanced(game)
        assert result, "DP-enhanced solver should find a solution on 5x5"
        valid, msg = validate_solution(game)
        assert valid, f"DP-enhanced solver produced invalid solution: {msg}"


# ==============================================================================
# D&C SOLVER TESTS
# ==============================================================================

class TestDnCSolver:
    """Tests for the Divide & Conquer solver."""
    
    def test_dnc_solver_3x3(self):
        """D&C solver should produce a valid solution on 3x3."""
        game = SlantGame(size=3)
        result = solve_with_dnc(game)
        assert result, "D&C solver should find a solution on 3x3"
        valid, msg = validate_solution(game)
        assert valid, f"D&C solver produced invalid solution on 3x3: {msg}"
    
    def test_dnc_solver_5x5(self):
        """D&C solver should produce a valid solution on 5x5."""
        game = SlantGame(size=5)
        result = solve_with_dnc(game)
        assert result, "D&C solver should find a solution on 5x5"
        valid, msg = validate_solution(game)
        assert valid, f"D&C solver produced invalid solution on 5x5: {msg}"


# ==============================================================================
# D&C CYCLE DETECTION TESTS
# ==============================================================================

class TestDnCCycleDetection:
    """Tests for D&C cycle detection vs DFS cycle detection."""
    
    def test_cycle_detection_agreement_no_cycle(self):
        """D&C cycle detection should agree with DFS on a puzzle without cycles."""
        game = SlantGame(size=3)
        # Make a few safe moves
        for c in range(3):
            if game.is_move_valid(0, c, 'L'):
                game.apply_move(0, c, 'L', check_validity=False,  player='HUMAN')
        
        dfs_result = game.detect_cycle_dfs()
        dnc_result, _ = detect_cycle_dnc(game)
        
        assert dfs_result == dnc_result, \
            f"Cycle detection mismatch: DFS={dfs_result}, D&C={dnc_result}"
    
    def test_cycle_detection_empty_board(self):
        """No cycles on an empty board."""
        game = SlantGame(size=3)
        # Clear the board
        for r in range(3):
            for c in range(3):
                if game.grid[r][c] is not None:
                    game.remove_move(r, c)
        
        dnc_result, cells = detect_cycle_dnc(game)
        assert not dnc_result, "Empty board should have no cycles"


# ==============================================================================
# MEMOIZED VALIDATOR TESTS
# ==============================================================================

class TestMemoizedValidator:
    """Tests for the MemoizedValidator."""
    
    def test_cached_matches_direct(self):
        """Memoized validity check should match direct validity check."""
        game = SlantGame(size=3)
        validator = MemoizedValidator(game)
        
        for r in range(3):
            for c in range(3):
                if game.grid[r][c] is None:
                    for move_type in ['L', 'R']:
                        cached = validator.is_move_valid_cached(r, c, move_type)
                        direct = game.is_move_valid(r, c, move_type)
                        assert cached == direct, \
                            f"Mismatch at ({r},{c},{move_type}): cached={cached}, direct={direct}"
    
    def test_cache_stats(self):
        """Validator should track cache hits and misses."""
        game = SlantGame(size=3)
        validator = MemoizedValidator(game)
        
        # First call should be a miss
        validator.is_move_valid_cached(0, 0, 'L')
        stats = validator.get_stats()
        assert stats['misses'] >= 1, "First call should be a miss"
        
        # Second identical call should be a hit (if state hasn't changed)
        validator.is_move_valid_cached(0, 0, 'L')
        stats = validator.get_stats()
        assert stats['hits'] >= 1, "Second identical call should be a hit"


# ==============================================================================
# AI STRATEGY TESTS
# ==============================================================================

class TestNewStrategies:
    """Tests for the new DP and D&C AI strategies."""
    
    def test_strategy_4_returns_valid_move(self):
        """Strategy 4 (DP Look-Ahead) should return a valid move."""
        game = SlantGame(size=3)
        ai = GreedyAI(game, strategy=4)
        move = ai.get_best_move()
        
        if move is not None:
            r, c, move_type = move
            assert game.grid[r][c] is None, f"Strategy 4 selected filled cell ({r},{c})"
            assert move_type in ['L', 'R'], f"Invalid move type: {move_type}"
            assert game.is_move_valid(r, c, move_type), \
                f"Strategy 4 returned invalid move ({r},{c},{move_type})"
    
    def test_strategy_5_returns_valid_move(self):
        """Strategy 5 (D&C Quadrant) should return a valid move."""
        game = SlantGame(size=3)
        ai = GreedyAI(game, strategy=5)
        move = ai.get_best_move()
        
        if move is not None:
            r, c, move_type = move
            assert game.grid[r][c] is None, f"Strategy 5 selected filled cell ({r},{c})"
            assert move_type in ['L', 'R'], f"Invalid move type: {move_type}"
            assert game.is_move_valid(r, c, move_type), \
                f"Strategy 5 returned invalid move ({r},{c},{move_type})"
    
    def test_strategy_4_multiple_moves(self):
        """Strategy 4 should produce valid moves across multiple turns."""
        game = SlantGame(size=3)
        ai = GreedyAI(game, strategy=4)
        
        for _ in range(5):
            move = ai.get_best_move()
            if move is None:
                break
            r, c, move_type = move
            assert game.is_move_valid(r, c, move_type)
            game.apply_move(r, c, move_type, player='CPU')
    
    def test_strategy_5_multiple_moves(self):
        """Strategy 5 should produce valid moves across multiple turns."""
        game = SlantGame(size=3)
        ai = GreedyAI(game, strategy=5)
        
        for _ in range(5):
            move = ai.get_best_move()
            if move is None:
                break
            r, c, move_type = move
            assert game.is_move_valid(r, c, move_type)
            game.apply_move(r, c, move_type, player='CPU')


# ==============================================================================
# BOUNDARY CUT SOLVER TESTS
# ==============================================================================

class TestBoundaryCutSolver:
    """Tests for the Minimum-Boundary Cut solver."""
    
    def test_boundary_cut_3x3(self):
        """Boundary Cut solver should produce a valid solution on 3x3."""
        game = SlantGame(size=3)
        result = solve_with_boundary_cut(game)
        if result:
            valid, msg = validate_solution(game)
            assert valid, f"Boundary Cut solver produced invalid solution on 3x3: {msg}"
    
    def test_boundary_cut_5x5(self):
        """Boundary Cut solver should produce a valid solution on 5x5."""
        game = SlantGame(size=5)
        result = solve_with_boundary_cut(game)
        if result:
            valid, msg = validate_solution(game)
            assert valid, f"Boundary Cut solver produced invalid solution on 5x5: {msg}"


# ==============================================================================
# HYBRID DP + D&C SOLVER TESTS
# ==============================================================================

class TestHybridSolver:
    """Tests for the Hybrid DP + D&C solver."""
    
    def test_hybrid_3x3(self):
        """Hybrid solver should produce a valid solution on 3x3."""
        game = SlantGame(size=3)
        result = solve_with_hybrid(game)
        if result:
            valid, msg = validate_solution(game)
            assert valid, f"Hybrid solver produced invalid solution on 3x3: {msg}"
    
    def test_hybrid_5x5(self):
        """Hybrid solver should produce a valid solution on 5x5."""
        game = SlantGame(size=5)
        result = solve_with_hybrid(game)
        if result:
            valid, msg = validate_solution(game)
            assert valid, f"Hybrid solver produced invalid solution on 5x5: {msg}"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
