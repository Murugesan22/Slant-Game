import random
from flask import Flask, jsonify, request
from flask_cors import CORS
from game_logic import SlantGame
from cpu_ai import AlgorithmicAI
from dp_solver import solve_with_dp, solve_with_dp_enhanced
from dnc_solver import solve_with_dnc, detect_cycle_dnc, solve_with_hybrid
from mrv_solver import solve_with_mrv
from cbj_solver import solve_with_cbj
from fc_solver import solve_with_fc
from mrv_solver import solve_with_mrv_recorded, get_mrv_stats, solve_partial_with_mrv_recorded
from cbj_solver import solve_with_cbj_recorded, get_cbj_stats
from fc_solver import solve_with_fc_recorded, get_fc_stats

app = Flask(__name__)
CORS(app) # Enable CORS for frontend

game = SlantGame(size=5)
ai_instance = None  # Global AI instance for multiplayer mode

# Solver system: maps solver type to configuration
SOLVER_CONFIG = {
    'dp': {'label': 'Dynamic Programming', 'description': 'Row-by-row DP with memoization'},
    'dnc': {'label': 'Divide & Conquer', 'description': 'Quadrant-based recursive solving'},
    'hybrid': {'label': 'Hybrid (D&C + DP)', 'description': 'D&C partitioning with DP solving'},
    'mrv': {'label': 'MRV Backtracking', 'description': 'Fills most constrained cell first'},
    'cbj': {'label': 'Conflict-Directed Backjumping', 'description': 'Jumps to conflict source on failure'},
    'fc':  {'label': 'Forward Checking', 'description': 'Prunes neighbour domains after each assignment'},
}
selected_solver = 'dp'  # Default solver type

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(game.to_dict())

@app.route('/api/new_game', methods=['POST'])
def new_game():
    global game, ai_instance
    data = request.json or {}
    size = data.get('size', 5)
    game = SlantGame(size=size)
    ai_instance = None  # Reset AI so it's rebuilt for the new board
    return jsonify(game.to_dict())

@app.route('/api/move', methods=['POST'])
def make_move():
    # Human move
    data = request.json
    r = data.get('row')
    c = data.get('col')
    move_type = data.get('type') # 'L', 'R', or None/CLEAR
    
    if r is None or c is None:
        return jsonify({"error": "Invalid params"}), 400
    
    # Handle "CLEAR" string from frontend if used
    if move_type == "CLEAR":
        move_type = None

    # Relaxed Rule: Allow Human to make invalid moves (check_validity=False)
    # The frontend will show visual errors (red markers).
    success = game.apply_move(r, c, move_type, check_validity=False, player='HUMAN')
    if not success:
        return jsonify({"error": "Invalid move", "state": game.to_dict()}), 400
    
    # If the board is now full but invalid, try to auto-fix by flipping CPU cells
    if game.status == "FILLED_INVALID":
        game.detect_cycle_dfs()  # Refresh loop_cells
        fixed = _try_fix_invalid_board(game)
        
        if fixed:
            return jsonify({
                "success": True,
                "auto_fixed": True,
                "message": "CPU auto-corrected its moves to reach a valid solution!",
                "state": game.to_dict()
            })
        else:
            return jsonify({
                "success": True,
                "user_fix_needed": True,
                "message": "Board is invalid. CPU corrected all its moves but couldn't fix it — please undo and change some of your moves.",
                "state": game.to_dict()
            })
    
    return jsonify({
        "success": True,
        "state": game.to_dict()
    })

@app.route('/api/cpu_move', methods=['POST'])
def cpu_move():
    global ai_instance, selected_solver
    if game.turn != 'CPU':
        return jsonify({"success": False, "message": "Not CPU turn", "state": game.to_dict()})

    config = SOLVER_CONFIG.get(selected_solver, SOLVER_CONFIG['dp'])
    solver_label = config['label']
    
    # Initialize AI if not already done
    if ai_instance is None:
        ai_instance = AlgorithmicAI(game, selected_solver)
    
    # Check if user deviated from solution, recompute if needed
    ai_instance.recompute_if_deviated()
    
    # Get next move — CPU always returns a move (forces 'L' if no solution)
    move = ai_instance.get_next_move()
    
    if move:
        cr, cc, ctype = move
        game.apply_move(cr, cc, ctype, check_validity=False, player='CPU')
        
        # If board is full but INVALID, signal to frontend for backtrack option
        if game.status == "FILLED_INVALID":
            game.detect_cycle_dfs()  # Refresh loop_cells for visual feedback
            return jsonify({
                "success": True,
                "cpu_move": {"row": cr, "col": cc, "type": ctype},
                "user_fix_needed": True,
                "message": f"Board filled with invalid moves. Use Backtrack & Fix to solve!",
                "state": game.to_dict()
            })
        
        return jsonify({
            "success": True, 
            "cpu_move": {"row": cr, "col": cc, "type": ctype},
            "state": game.to_dict()
        })
    else:
        # Board is already full
        game.turn = 'HUMAN'
        return jsonify({
            "success": True,
            "message": "Board is full.",
            "state": game.to_dict()
        })


def _try_fix_invalid_board(game):
    """
    When the board is full but invalid (cycles or constraint violations),
    try to fix it by flipping CPU-owned cells that are involved in cycles.
    Uses backtracking to find a valid combination of flips.
    Returns True if fixed, False otherwise.
    """
    # Identify CPU-owned cells that are in cycles
    loop_cells = getattr(game, 'loop_cells', [])
    print(f"[AUTO-FIX] Board is FILLED_INVALID. loop_cells={loop_cells}")
    
    # Collect ALL CPU-owned cells (potential fix targets)
    cpu_cells = []
    for r in range(game.size):
        for c in range(game.size):
            if game.owners[r][c] == 'CPU':
                cpu_cells.append((r, c))
    
    print(f"[AUTO-FIX] CPU owns {len(cpu_cells)} cells: {cpu_cells}")
    
    # Prioritize CPU cells that are in cycles
    cycle_set = set((lc[0], lc[1]) if isinstance(lc, (list, tuple)) else lc for lc in loop_cells)
    cpu_cycle_cells = [cell for cell in cpu_cells if cell in cycle_set]
    cpu_other_cells = [cell for cell in cpu_cells if cell not in cycle_set]
    
    print(f"[AUTO-FIX] CPU cells in cycles: {cpu_cycle_cells}, others: {cpu_other_cells}")
    
    # Try fixing cycle cells first, then other CPU cells if needed
    fix_candidates = cpu_cycle_cells + cpu_other_cells
    
    if not fix_candidates:
        print("[AUTO-FIX] No CPU cells to fix — all moves are human")
        return False
    
    # Backtracking: try flipping subsets of CPU cells to fix the board
    result = _backtrack_fix(game, fix_candidates, 0)
    print(f"[AUTO-FIX] Backtrack result: {result}, final status: {game.status}")
    return result


def _backtrack_fix(game, candidates, idx):
    """
    Backtracking to flip CPU cells and find a valid board state.
    Tries flipping each candidate cell and checks if the board becomes valid.
    """
    # Check if current state is valid
    game.check_completion()
    if game.status in ("WIN_HUMAN", "WIN_CPU", "DRAW", "COMPLETED"):
        return True  # Board is now valid!
    
    if idx >= len(candidates):
        return False  # Exhausted all candidates
    
    r, c = candidates[idx]
    current_val = game.grid[r][c]
    alt_val = 'R' if current_val == 'L' else 'L'
    
    # Try flipping this cell
    # Remove old move
    game.remove_move(r, c)
    
    # Apply alternative
    game.grid[r][c] = alt_val
    if alt_val == 'L':
        n1, n2 = (r, c), (r+1, c+1)
        game._add_edge(n1, n2)
    else:
        n1, n2 = (r+1, c), (r, c+1)
        game._add_edge(n1, n2)
    game.node_degrees[n1] += 1
    game.node_degrees[n2] += 1
    
    game.check_completion()
    game.detect_cycle_dfs()
    
    if game.status in ("WIN_HUMAN", "WIN_CPU", "DRAW", "COMPLETED"):
        return True
    
    # Recurse: try fixing more cells
    if _backtrack_fix(game, candidates, idx + 1):
        return True
    
    # Undo the flip — restore original value
    game.remove_move(r, c)
    game.grid[r][c] = current_val
    if current_val == 'L':
        n1, n2 = (r, c), (r+1, c+1)
        game._add_edge(n1, n2)
    else:
        n1, n2 = (r+1, c), (r, c+1)
        game._add_edge(n1, n2)
    game.node_degrees[n1] += 1
    game.node_degrees[n2] += 1
    
    # Try without flipping this cell — skip to next
    if _backtrack_fix(game, candidates, idx + 1):
        return True
    
    return False

@app.route('/api/undo', methods=['POST'])
def undo_move():
    if game.undo():
        return jsonify({"success": True, "state": game.to_dict()})
    else:
        return jsonify({"error": "Nothing to undo", "state": game.to_dict()}), 400

@app.route('/api/backtrack_fix', methods=['POST'])
def backtrack_fix():
    """
    Fix the invalid board by clearing only the bad cells (cycles + constraint
    violations) and running MRV backtracking to fill them. Records steps for
    animated visualization. Only fixes wrong moves — keeps correct ones.
    If the partial board is unsolvable (e.g. user manually changed cells that
    conflict with every valid solution), falls back to clearing ALL cells and
    solving from scratch so the MRV solver always has a viable starting state.
    """
    global game
    try:
        import copy
        game_copy = copy.deepcopy(game)
        size = game_copy.size
        
        # Snapshot scores before fixing — backtrack correction must NOT change scores
        score_snapshot = {k: v for k, v in game.scores.items()}
        # Identify bad cells: cells in loops + cells adjacent to violated constraints
        bad_cells = _get_bad_cells(game_copy)
        
        if not bad_cells:
            # No bad cells found but board is invalid — fall back to clearing all
            for r in range(size):
                for c in range(size):
                    if game_copy.grid[r][c] is not None:
                        game_copy.remove_move(r, c)
            game_copy.history = []
            game_copy.scores = {'HUMAN': 0, 'CPU': 0}
            game_copy.owners = [[None for _ in range(size)] for _ in range(size)]
        else:
            # Clear only the bad cells
            for (r, c) in bad_cells:
                if game_copy.grid[r][c] is not None:
                    game_copy.remove_move(r, c)
            game_copy.history = []
        
        # Run MRV solver on the partial board (only fills empty cells)
        success, steps, solved_copy = solve_partial_with_mrv_recorded(game_copy)
        stats = get_mrv_stats(steps)
        final_grid = solved_copy.grid if success else None
        # --- Fallback: partial board is unsolvable ---
        # This happens when the user manually changed cells that are incompatible
        # with every valid solution. Clear ALL cells and solve from a clean state.
        if not success:
            full_copy = copy.deepcopy(game)
            for r in range(size):
                for c in range(size):
                    if full_copy.grid[r][c] is not None:
                        full_copy.remove_move(r, c)
            full_copy.history = []
            full_copy.scores = {'HUMAN': 0, 'CPU': 0}
            full_copy.owners = [[None for _ in range(size)] for _ in range(size)]
            success, steps, solved_copy = solve_partial_with_mrv_recorded(full_copy)
            stats = get_mrv_stats(steps)
            final_grid = solved_copy.grid if success else None
            # Mark every cell as "bad" since we cleared the whole board
            bad_cells = set((r, c) for r in range(size) for c in range(size))
        if success:
            # Apply solution to the real game: only update the cells that differ
            for r in range(size):
                for c in range(size):
                    old_mv = game.grid[r][c]
                    new_mv = solved_copy.grid[r][c]
                    if old_mv != new_mv:
                        if old_mv is not None:
                            game.remove_move(r, c)
                        if new_mv is not None:
                            game.apply_move(r, c, new_mv, check_validity=False, player='HUMAN')
            # Clear stale loop_cells BEFORE check_completion so the frontend
            # does not render red cells on a successfully fixed board.
            game.loop_cells = []
            game.check_completion()
            # Re-run cycle detection to finalise loop_cells for the solved state.
            # For a valid solution this always produces [] — no red cells.
            game.detect_cycle_dfs()
            # Restore scores — corrections must not award points
            game.scores = score_snapshot
        
        return jsonify({
            "success": success,
            "steps": steps,
            "stats": stats,
            # bad_cells may be a set of tuples after the fallback — serialise correctly
            "bad_cells": [list(bc) for bc in bad_cells] if bad_cells else [],
            "final_grid": final_grid,
            "state": game.to_dict(),
            "message": f"Fixed {len(bad_cells)} bad cells: {len(steps)} steps, {stats.get('backtracks', 0)} backtracks"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Backtrack fix error: {str(e)}"
        }), 500


def _get_bad_cells(game):
    """
    Identify cells that are causing the board to be invalid:
    1. Cells in loops (cycle detection)
    2. Cells adjacent to violated constraints
    """
    bad = set()
    size = game.size
    
    # 1. Cells in loops
    game.detect_cycle_dfs()
    if hasattr(game, 'loop_cells') and game.loop_cells:
        for (r, c) in game.loop_cells:
            if 0 <= r < size and 0 <= c < size:
                bad.add((r, c))
    
    # 2. Cells adjacent to violated constraints
    for node, limit in game.constraints.items():
        actual = game.node_degrees.get(node, 0)
        if actual != limit:
            nr, nc = node
            # Each node (nr, nc) is a corner shared by up to 4 cells
            for dr, dc in [(-1, -1), (-1, 0), (0, -1), (0, 0)]:
                cr, cc = nr + dr, nc + dc
                if 0 <= cr < size and 0 <= cc < size and game.grid[cr][cc] is not None:
                    bad.add((cr, cc))
    
    return bad

@app.route('/api/set_solver', methods=['POST'])
def set_solver():
    global selected_solver, ai_instance
    data = request.json or {}
    
    solver = data.get('solver', 'dp')
    if solver not in SOLVER_CONFIG:
        return jsonify({"error": "Invalid solver. Must be 'dp', 'dnc', 'hybrid', 'mrv', 'cbj', or 'fc'"}), 400
    
    selected_solver = solver
    # Reset AI instance to force recomputation with new solver
    ai_instance = None
    
    return jsonify({
        "success": True, 
        "solver": selected_solver,
        "label": SOLVER_CONFIG[solver]['label'],
        "description": SOLVER_CONFIG[solver]['description']
    })

@app.route('/api/solve', methods=['POST'])
def solve_game():
    """Solve using the currently selected algorithmic solver."""
    global selected_solver
    
    # Route to appropriate solver based on selection
    if selected_solver == 'dp':
        return solve_dp()
    elif selected_solver == 'dnc':
        return solve_dnc()
    elif selected_solver == 'hybrid':
        return solve_hybrid()

    elif selected_solver == 'mrv':
        return solve_mrv()
    elif selected_solver == 'cbj':
        return solve_cbj()
    elif selected_solver == 'fc':
        return solve_fc()
    else:
        # Default to DP
        return solve_dp()

@app.route('/api/solve_dp', methods=['POST'])
def solve_dp():
    """Solve using Dynamic Programming approach."""
    global game
    try:
        # Try Profile DP first, fall back to DP-enhanced backtracking
        success = solve_with_dp(game)
        if not success:
            # Reset and try DP-enhanced backtracking
            game._initialize_empty_state()
            success = solve_with_dp_enhanced(game)
        
        if success:
            return jsonify({
                "success": True, 
                "state": game.to_dict(), 
                "message": "Solved using Dynamic Programming!",
                "algorithm": "dp"
            })
        else:
            return jsonify({
                "success": False, 
                "state": game.to_dict(), 
                "message": "DP solver could not find a solution"
            }), 400
    except Exception as e:
        return jsonify({
            "success": False, 
            "state": game.to_dict(), 
            "message": f"DP solver error: {str(e)}"
        }), 500

@app.route('/api/solve_dnc', methods=['POST'])
def solve_dnc():
    """Solve using Divide and Conquer approach."""
    global game
    try:
        success = solve_with_dnc(game)
        
        if success:
            return jsonify({
                "success": True, 
                "state": game.to_dict(), 
                "message": "Solved using Divide & Conquer!",
                "algorithm": "dnc"
            })
        else:
            return jsonify({
                "success": False, 
                "state": game.to_dict(), 
                "message": "D&C solver could not find a solution"
            }), 400
    except Exception as e:
        return jsonify({
            "success": False, 
            "state": game.to_dict(), 
            "message": f"D&C solver error: {str(e)}"
        }), 500



@app.route('/api/solve_hybrid', methods=['POST'])
def solve_hybrid():
    """Solve using Hybrid DP + D&C approach."""
    global game
    try:
        success = solve_with_hybrid(game)
        
        if success:
            return jsonify({
                "success": True, 
                "state": game.to_dict(), 
                "message": "Solved using Hybrid DP + D&C!",
                "algorithm": "hybrid"
            })
        else:
            return jsonify({
                "success": False, 
                "state": game.to_dict(), 
                "message": "Hybrid solver could not find a solution"
            }), 400
    except Exception as e:
        return jsonify({
            "success": False, 
            "state": game.to_dict(), 
            "message": f"Hybrid solver error: {str(e)}"
        }), 500

@app.route('/api/solve_mrv', methods=['POST'])
def solve_mrv():
    """Solve using MRV Backtracking approach."""
    global game
    try:
        success = solve_with_mrv(game)

        if success:
            return jsonify({
                "success": True,
                "state": game.to_dict(),
                "message": "Solved using MRV Backtracking!",
                "algorithm": "mrv"
            })
        else:
            return jsonify({
                "success": False,
                "state": game.to_dict(),
                "message": "MRV solver could not find a solution"
            }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "state": game.to_dict(),
            "message": f"MRV solver error: {str(e)}"
        }), 500

@app.route('/api/solve_cbj', methods=['POST'])
def solve_cbj():
    """Solve using Conflict-Directed Backjumping approach."""
    global game
    try:
        success = solve_with_cbj(game)

        if success:
            return jsonify({
                "success": True,
                "state": game.to_dict(),
                "message": "Solved using Conflict-Directed Backjumping!",
                "algorithm": "cbj"
            })
        else:
            return jsonify({
                "success": False,
                "state": game.to_dict(),
                "message": "CBJ solver could not find a solution"
            }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "state": game.to_dict(),
            "message": f"CBJ solver error: {str(e)}"
        }), 500

@app.route('/api/solve_fc', methods=['POST'])
def solve_fc():
    """Solve using Forward Checking approach."""
    global game
    try:
        success = solve_with_fc(game)

        if success:
            return jsonify({
                "success": True,
                "state": game.to_dict(),
                "message": "Solved using Forward Checking!",
                "algorithm": "fc"
            })
        else:
            return jsonify({
                "success": False,
                "state": game.to_dict(),
                "message": "FC solver could not find a solution"
            }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "state": game.to_dict(),
            "message": f"FC solver error: {str(e)}"
        }), 500

@app.route('/api/visualize', methods=['POST'])
def visualize_solver():
    """Run a solver with step recording for visualization."""
    global game
    data = request.json or {}
    solver = data.get('solver', 'mrv')
    try:
        import copy
        game_copy = copy.deepcopy(game)
        if solver == 'mrv':
            success, steps = solve_with_mrv_recorded(game_copy)
            stats = get_mrv_stats(steps)
            label = 'MRV Backtracking'
        elif solver == 'cbj':
            success, steps = solve_with_cbj_recorded(game_copy)
            stats = get_cbj_stats(steps)
            label = 'Conflict-Directed Backjumping'
        elif solver == 'fc':
            success, steps = solve_with_fc_recorded(game_copy)
            stats = get_fc_stats(steps)
            label = 'Forward Checking'
        else:
            return jsonify({"error": "solver must be 'mrv', 'cbj', or 'fc'"}), 400
        final_grid = game_copy.grid if success else None
        return jsonify({
            "success": success,
            "solver": solver,
            "label": label,
            "steps": steps,
            "stats": stats,
            "final_grid": final_grid,
            "message": f"Recorded {len(steps)} steps using {label}"
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Visualization error: {str(e)}"}), 500

@app.route('/api/review', methods=['POST'])
def review_moves():
    """
    Review all moves against the solution grid.
    Returns list of all moves with correctness status.
    """
    global game, ai_instance
    
    try:
        # If solution_grid not set, compute it using current solver
        if not game.solution_grid:
            # Use AI to compute solution
            if ai_instance is None:
                ai_instance = AlgorithmicAI(game, selected_solver)
            
            if not ai_instance.compute_solution():
                return jsonify({
                    "success": False,
                    "message": "Could not compute solution for review"
                }), 400
            
            # Store the solution grid
            game.solution_grid = ai_instance.solution_grid
        
        # Get all move reviews (correct and incorrect)
        move_reviews = game.get_move_reviews()
        
        # Count incorrect moves
        incorrect_count = sum(1 for move in move_reviews if not move['correct'])
        
        return jsonify({
            "success": True,
            "move_reviews": move_reviews,
            "total_moves": len(move_reviews),
            "incorrect_count": incorrect_count
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Review error: {str(e)}"
        }), 500

@app.route('/api/compare_solvers', methods=['POST'])
def compare_solvers():
    """
    Benchmark all solvers on the current puzzle and return time + space data
    for comparison charts.
    """
    import copy
    import time
    import sys
    
    size = game.size
    results = []
    
    solvers = [
        ('dp', 'Dynamic Programming', lambda g: solve_with_dp(g) or solve_with_dp_enhanced(g)),
        ('dnc', 'Divide & Conquer', lambda g: solve_with_dnc(g)),
        ('hybrid', 'Hybrid (D&C + DP)', lambda g: solve_with_hybrid(g)),
        ('mrv', 'MRV Backtracking', lambda g: solve_with_mrv(g)),
        ('cbj', 'Conflict Backjumping', lambda g: solve_with_cbj(g)),
        ('fc',  'Forward Checking', lambda g: solve_with_fc(g)),
    ]
    
    for key, label, solver_fn in solvers:
        game_copy = copy.deepcopy(game)
        # Clear the copy
        for r in range(size):
            for c in range(size):
                if game_copy.grid[r][c] is not None:
                    game_copy.remove_move(r, c)
        game_copy.history = []
        game_copy.scores = {'HUMAN': 0, 'CPU': 0}
        game_copy.owners = [[None for _ in range(size)] for _ in range(size)]
        
        try:
            start = time.perf_counter()
            success = solver_fn(game_copy)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            # Estimate space: recursion depth * state size
            space_estimates = {
                'dp': size * size * (2 ** size),        # DP table
                'dnc': size * size * 4,                  # Quadrant recursion
                'hybrid': size * size * (2 ** (size//2)), # Hybrid
                'mrv': size * size * size,                # MRV stack depth
                'cbj': size * size * size * 2,            # CBJ + conflict sets
                'fc':  size * size * size * 2,            # FC + domain snapshots
            }
            
            results.append({
                'key': key,
                'label': label,
                'time_ms': round(elapsed, 2),
                'space': space_estimates.get(key, size * size),
                'success': success
            })
        except Exception as e:
            results.append({
                'key': key,
                'label': label,
                'time_ms': -1,
                'space': 0,
                'success': False,
                'error': str(e)
            })
    
    return jsonify({
        "success": True,
        "results": results,
        "board_size": size
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
