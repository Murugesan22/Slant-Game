import random
from flask import Flask, jsonify, request
from flask_cors import CORS
from game_logic import SlantGame
from cpu_ai import AlgorithmicAI
from dp_solver import solve_with_dp, solve_with_dp_enhanced
from dnc_solver import solve_with_dnc, detect_cycle_dnc, solve_with_boundary_cut, solve_with_hybrid

app = Flask(__name__)
CORS(app) # Enable CORS for frontend

game = SlantGame(size=5)
ai_instance = None  # Global AI instance for multiplayer mode

# Solver system: maps solver type to configuration
SOLVER_CONFIG = {
    'dp': {'label': 'Dynamic Programming', 'description': 'Row-by-row DP with memoization'},
    'dnc': {'label': 'Divide & Conquer', 'description': 'Quadrant-based recursive solving'},
    'hybrid': {'label': 'Hybrid (D&C + DP)', 'description': 'D&C partitioning with DP solving'},
    'cut': {'label': 'Cut-based Partition', 'description': 'Minimal cut region solving'},
}
selected_solver = 'dp'  # Default solver type

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(game.to_dict())

@app.route('/api/new_game', methods=['POST'])
def new_game():
    global game
    data = request.json or {}
    size = data.get('size', 5)
    game = SlantGame(size=size)
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
    
    # Get next move from precomputed solution
    move = ai_instance.get_next_move()
    
    if move:
        cr, cc, ctype = move
        game.apply_move(cr, cc, ctype, player='CPU')
        
        # If board is full but INVALID, attempt to auto-fix CPU moves
        if game.status == "FILLED_INVALID":
            fixed = _try_fix_invalid_board(game)
            
            if fixed:
                return jsonify({
                    "success": True,
                    "cpu_move": {"row": cr, "col": cc, "type": ctype},
                    "message": f"CPU auto-corrected its moves using {solver_label} algorithm",
                    "auto_fixed": True,
                    "state": game.to_dict()
                })
            else:
                # CPU exhausted all its moves — problem is with human moves
                return jsonify({
                    "success": False,
                    "user_fix_needed": True,
                    "message": f"CPU corrected all its moves but the board is still invalid. Please change some of your moves to reach a valid solution.",
                    "solver": selected_solver,
                    "state": game.to_dict()
                })
        
        return jsonify({
            "success": True, 
            "cpu_move": {"row": cr, "col": cc, "type": ctype},
            "state": game.to_dict()
        })
    else:
        # CPU Pass - no solution found
        game.turn = 'HUMAN'
        return jsonify({"success": True, "message": "CPU Passed (No valid moves found)", "state": game.to_dict()})


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

@app.route('/api/set_solver', methods=['POST'])
def set_solver():
    global selected_solver, ai_instance
    data = request.json or {}
    
    solver = data.get('solver', 'dp')
    if solver not in SOLVER_CONFIG:
        return jsonify({"error": "Invalid solver. Must be 'dp', 'dnc', 'hybrid', or 'cut'"}), 400
    
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
    elif selected_solver == 'cut':
        return solve_boundary_cut()
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

@app.route('/api/solve_boundary_cut', methods=['POST'])
def solve_boundary_cut():
    """Solve using Minimum-Boundary Cut D&C approach."""
    global game
    try:
        success = solve_with_boundary_cut(game)
        
        if success:
            return jsonify({
                "success": True, 
                "state": game.to_dict(), 
                "message": "Solved using Minimum-Boundary Cut!",
                "algorithm": "boundary_cut"
            })
        else:
            return jsonify({
                "success": False, 
                "state": game.to_dict(), 
                "message": "Boundary Cut solver could not find a solution"
            }), 400
    except Exception as e:
        return jsonify({
            "success": False, 
            "state": game.to_dict(), 
            "message": f"Boundary Cut solver error: {str(e)}"
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
