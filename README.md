# 🎮 SLANT — Interactive Puzzle Game

> A graph theory–based constraint-satisfaction puzzle with **six solving algorithms**, real-time **step-by-step visualization**, **multiplayer AI**, and **comparative performance analysis**.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Game Rules](#game-rules)
- [How to Play](#how-to-play)
- [Game Modes](#game-modes)
- [Scoring System](#scoring-system)
- [Solving Algorithms (Strategies)](#solving-algorithms-strategies)
  - [1. Dynamic Programming (DP)](#1-dynamic-programming-dp)
  - [2. Divide & Conquer (D&C)](#2-divide--conquer-dc)
  - [3. Hybrid DP + D&C](#3-hybrid-dp--dc)
  - [4. MRV Backtracking](#4-mrv-backtracking)
  - [5. Conflict-Directed Backjumping (CBJ)](#5-conflict-directed-backjumping-cbj)
  - [6. Forward Checking (FC)](#6-forward-checking-fc)
- [CPU AI Strategies (Multiplayer)](#cpu-ai-strategies-multiplayer)
- [Key Features](#key-features)
  - [Backtracking Visualization](#backtracking-visualization)
  - [Comparative Analysis Dashboard](#comparative-analysis-dashboard)
  - [Backtrack & Fix](#backtrack--fix)
  - [Move Review](#move-review)
- [Complexity Analysis](#complexity-analysis)
- [Tech Stack & Architecture](#tech-stack--architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)

---

## Overview

**SLANT** is an educational and competitive puzzle game built to demonstrate core **Design and Analysis of Algorithms (DAA)** concepts through interactive gameplay. Players fill an N×N grid with diagonal slashes (`\` or `/`) while satisfying numbered constraints at grid intersections and avoiding loops (cycles) in the resulting edge graph.

The project implements **six distinct algorithmic solving strategies** spanning Dynamic Programming, Divide & Conquer, Backtracking with Heuristics, Constraint Propagation, and Conflict-Directed Backjumping — each with real-time visualization and performance benchmarking.

### Key Highlights

| Feature | Description |
|---------|-------------|
| **6 Solving Algorithms** | DP, D&C, Hybrid, MRV, CBJ, Forward Checking |
| **Step-by-Step Visualizer** | Animated playback for MRV, CBJ, and FC solvers |
| **Performance Analysis** | Comparative bar charts, line charts, and Big-O complexity graphs |
| **Multiplayer Mode** | Compete against 5 CPU AI strategies with 3 difficulty levels |
| **Variable Board Sizes** | 3×3, 5×5, 7×7, 9×9 grids |
| **Interactive UI** | Sound effects, animations, color-coded feedback, responsive design |

---

## Game Rules

### Objective

Complete the grid by filling every cell with a diagonal slash while satisfying all constraints and creating no loops.

### Three Core Rules

#### 1️⃣ Fill Every Cell
Every cell must contain exactly one diagonal slash:
- **Left Slash (`\`)**: Connects top-left corner → bottom-right corner
- **Right Slash (`/`)**: Connects top-right corner → bottom-left corner

#### 2️⃣ Satisfy Number Constraints
Intersection points (corners where cells meet) may have numbers (0–4) indicating exactly how many lines must touch that point.

```
     1   2   1
   ┌───┬───┬───┐
 1 │ \ │ / │ \ │ 1
   ├───┼───┼───┤
 2 │ / │ \ │ / │ 2
   ├───┼───┼───┤
 1 │ \ │ / │ \ │ 1
   └───┴───┴───┘
     1   2   1
```

#### 3️⃣ No Loops Allowed
Slashes must **not** form closed loops/cycles. Lines can branch like a tree but cannot circle back.

---

## How to Play

### Controls

| Action | Input |
|--------|-------|
| Place / toggle slash | Single click on cell |
| Clear a cell | Double-click on cell |
| New puzzle | **New Game** button |
| Undo last move | **Undo** button |
| Auto-solve | **Solve** button → pick algorithm |
| Visualize solver | **Visualize** button → pick solver |
| View analysis | **Analysis** button (after solving) |
| Review moves | **Review** button |

### Visual Feedback

| Color | Meaning |
|-------|---------|
| 🟢 Green number | Constraint satisfied |
| 🔴 Red number | Constraint violated |
| 🟡 Yellow number | Constraint partially fulfilled |
| 🔴 Red slashes | Part of a detected loop/cycle |
| 🔵 Blue highlight | CPU-placed move |

---

## Game Modes

### Single Player (Default)
Free-play puzzle-solving mode. Place slashes at your own pace with full undo support.

### Multiplayer (vs CPU)
Turn-based competitive mode. Click **Multiplayer** → select a CPU strategy → alternate turns. The player with the highest score at completion wins.

### Solve Mode
Select any of the 6 algorithms to auto-complete the puzzle. Demonstrates that a valid solution exists.

---

## Scoring System

Points are awarded identically to both human and CPU for fairness.

| Criterion | Points | Condition |
|-----------|--------|-----------|
| **Base Move** | +1 | Every valid move |
| **Constraint Satisfaction** | +2 per constraint | Move completes a numbered node's requirement |
| **Perfect Cell** | +3 | All 4 corners of the cell are fully satisfied |
| **Strategic Position** | +1 | Move placed within 1 cell of the board center |

**Maximum per move: 9 points** (base + 4 constraints + perfect cell + center bonus)

---

## Solving Algorithms (Strategies)

### 1. Dynamic Programming (DP)

**File:** `backend/dp_solver.py`

#### Profile DP Solver (`solve_with_dp`)
Processes the grid **row by row**, treating each row as a DP stage. The "profile" is the state of node degrees along the boundary between the current row and the next.

**Key Principles:**
- **Optimal Substructure**: If rows 0..k are filled correctly, remaining rows depend only on the boundary state at row k
- **Overlapping Subproblems**: Different row assignments can produce the same boundary state; memoization avoids re-solving

**How It Works:**
1. For each row, try all 2^N possible assignments (L or R per cell)
2. Capture the boundary state (tuple of node degrees along the bottom edge)
3. Memoize `(row, boundary_state)` — skip if already explored
4. After all rows, verify final constraints

#### DP-Enhanced Backtracking (`solve_with_dp_enhanced`)
Combines standard backtracking with:
- **Memoized Validity Checks**: Caches constraint + cycle detection results to avoid redundant BFS
- **Row-Boundary Pruning**: After completing each row, verify all constraints before proceeding

**Complexity:** `O(2^N · N² · S)` where S = number of unique boundary states

---

### 2. Divide & Conquer (D&C)

**File:** `backend/dnc_solver.py`

#### Grid Partitioning Solver (`solve_with_dnc`)
Recursively divides the N×N grid into **4 quadrants**, solves each, and combines while resolving boundary conflicts.

```
   ┌─────────────┬─────────────┐
   │  Top-Left   │  Top-Right  │
   │  Quadrant   │  Quadrant   │
   ├─────────────┼─────────────┤
   │ Bottom-Left │ Bottom-Right│
   │  Quadrant   │  Quadrant   │
   └─────────────┴─────────────┘
```

**Steps:**
1. **Divide**: Split cells into 4 quadrants by median row/column
2. **Conquer**: Solve each quadrant recursively (base case ≤ 4 cells)
3. **Combine**: Handle boundary nodes where quadrants share constraints; backtrack if a quadrant fill makes another unsolvable

#### D&C Cycle Detection (`detect_cycle_dnc`)
Detects cycles via spatial decomposition — split graph into halves, detect cycles in each, then check cross-boundary cycles using a contracted super-node graph.

**Complexity:** `O(4 · N²)` — linear in grid area

---

### 3. Hybrid DP + D&C

**File:** `backend/dnc_solver.py` → `solve_with_hybrid()`

Combines the best of both paradigms:

1. **Divide** (D&C): Split grid into top half and bottom half
2. **Conquer** (DP): Enumerate all valid boundary states for the top half using DP
3. **Combine**: For each boundary state, try solving the bottom half via backtracking

This transforms the boundary conflict problem from trial-and-error into a structured **set-intersection problem**.

**Complexity:** `O(2^(N/2) · N²)` — exponential but with a halved exponent

---

### 4. MRV Backtracking

**File:** `backend/mrv_solver.py`

Uses the **Minimum Remaining Values (MRV)** heuristic — always picks the most constrained empty cell first, dramatically reducing the search tree.

**Algorithm:**
1. Find the cell with the **fewest valid moves** (MRV heuristic)
2. Try each valid move ordered by constraint satisfaction score (most satisfying first)
3. Apply move → recurse → undo on failure (backtrack)
4. Verify complete solution: all cells filled, all constraints met, no cycles

**Why MRV Works:**
If a cell has only 1 valid option, it's placed immediately (forced move). Cells with 0 options trigger early backtracking. This is the same technique used by competitive Sudoku solvers.

**Complexity:** `O(2^(N²))` worst case, but MRV pruning makes it `O(N³)` in practice

---

### 5. Conflict-Directed Backjumping (CBJ)

**File:** `backend/cbj_solver.py`

Instead of chronological backtracking (undo one step at a time), CBJ identifies **which earlier decision caused the current failure** and jumps directly back to it.

**Algorithm:**
1. Maintain a **conflict set** for each cell — tracks which earlier cells' assignments are incompatible
2. When a cell has no valid moves → find the **conflict source** (the earliest cell that actually caused the failure)
3. **Jump** back to the conflict source, skipping intermediate cells entirely
4. Merge conflict sets when jumping past intermediate cells

**Two Conflict Types Detected:**
- **Degree violations**: A node's degree exceeds its constraint
- **Cycle creation**: A move would create a cycle; the conflicting cell is traced along the cycle path

**Advantage over plain backtracking:** Avoids re-exploring branches that fail for the same structural reason, skipping potentially thousands of fruitless states.

**Complexity:** `O(2^(N²))` worst case, `O(N² · log N)` in practice due to intelligent jumping

---

### 6. Forward Checking (FC)

**File:** `backend/fc_solver.py`

Uses **constraint propagation** after each assignment to proactively prune the domains of unassigned neighboring cells.

**Algorithm:**
1. Initialize domains: each empty cell has domain `{L, R}`
2. Pick the next cell, try each value in its domain
3. After placing a move, run **forward checking**:
   - For each unassigned neighbor, remove values that would cause immediate constraint violations or cycles
   - If any neighbor's domain becomes **empty** → **wipeout** → backtrack immediately
4. On backtrack, restore pruned domains

**Advantage:** Detects failures **before** reaching the failing cell, pruning the search tree earlier than standard backtracking.

**Complexity:** `O(2^(N²))` worst case, `O(2 · N³)` in practice due to aggressive domain pruning

---

## CPU AI Strategies (Multiplayer)

The CPU uses an `AlgorithmicAI` class with **5 greedy strategies** mapped to 3 difficulty levels:

| Difficulty | Strategy | Mistake Rate |
|------------|----------|-------------|
| Easy | Random-Greedy | 40% random moves |
| Medium | Constraint-Focused | 15% random moves |
| Hard | DP-Lookahead | 0% random moves |

### Strategy 1: Constraint-Focused
Prioritizes cells adjacent to the most constraints. Scores each move by constraint urgency and satisfaction potential. Cells where placing a slash immediately satisfies a constraint get heavy bonuses.

### Strategy 2: Edge-First
Fills cells from the **edges inward**. Edge/corner cells have fewer neighbors and are easier to determine correctly — mimics how human solvers approach the puzzle.

### Strategy 3: Random-Greedy
Uses basic constraint scoring but selects randomly among the top 3 moves. Creates unpredictable, weaker play for Easy difficulty.

### Strategy 4: DP-Lookahead (Hard)
The most sophisticated strategy. Simulates each candidate move, checks how it affects future solvability via **memoized validity caching**, and selects the move that maximizes neighbor freedom (avoids dead-ends).

### Strategy 5: D&C Quadrant-Based
Divides the board into 4 quadrants, calculates **constraint pressure** per quadrant, and prioritizes moves in the most constrained quadrant. Boundary cells shared between quadrants get bonuses.

### Move Sorting — Custom Quicksort
All CPU strategies use a custom **Quicksort** (D&C) implementation to sort candidate moves by heuristic score. Uses randomized pivot selection for `O(N log N)` average-case performance.

---

## Key Features

### Backtracking Visualization

Real-time, step-by-step animated playback for **MRV**, **CBJ**, and **FC** solvers. Watch the algorithm think:

| Highlight | Meaning |
|-----------|---------|
| 🟡 Yellow | Trying a move |
| 🟢 Green | Forced move (only option) |
| 🟠 Orange | Backtracking |
| 🔴 Red | Dead end |
| 🔵 Cyan | Successfully placed |

**Controls:** Play, Pause, Step, Reset, Speed slider.

**Statistics shown:**
- Total steps taken
- Backtracks count
- Forced moves count
- Skipped cells (CBJ/FC)
- Backjumps (CBJ only)

---

### Comparative Analysis Dashboard

Click **📊 Analysis** after solving to see three comparison charts:

1. **Time Chart (Line)** — Actual solve time (ms) for all 6 algorithms on the current puzzle
2. **Space Chart (Bar)** — Estimated memory usage for each algorithm
3. **Complexity Chart (Curves)** — Theoretical Big-O growth curves on a log scale:

| Algorithm | Time Complexity |
|-----------|----------------|
| DP | O(2^N · N²) |
| D&C | O(4 · N²) |
| Hybrid | O(2^(N/2) · N²) |
| MRV | O(N³) |
| CBJ | O(N² · log N) |
| FC | O(2 · N³) |

A vertical marker highlights the current board size `n` for reference.

---

### Backtrack & Fix

When the board is filled but **invalid** (cycles or constraint violations), click **Backtrack & Fix** to:
1. Identify bad cells (in loops or near violated constraints)
2. Clear only the bad cells
3. Run MRV backtracking to fill them correctly
4. Animate the entire fix process step by step

Falls back to a full-board re-solve if the partial board is unsolvable.

---

### Move Review

Click **📝 Review** to analyze all placed moves against the computed solution:
- ✔ Correct moves highlighted in green
- ✘ Incorrect moves highlighted in red
- Summary: `X incorrect / Y total moves`

---

## Complexity Analysis

### Solver Complexity Comparison

| Algorithm | Time (Worst) | Time (Practical) | Space |
|-----------|-------------|-------------------|-------|
| **DP** (Profile) | O(2^N · N² · S) | Fast for N ≤ 9 | O(N · S) memoization table |
| **D&C** (Partitioning) | O(4 · N²) | Very fast | O(N²) quadrant recursion |
| **Hybrid** (DP + D&C) | O(2^(N/2) · N²) | Moderate | O(2^(N/2)) boundary states |
| **MRV** | O(2^(N²)) | O(N³) via MRV pruning | O(N²) recursion stack |
| **CBJ** | O(2^(N²)) | O(N² · log N) via backjumping | O(N²) conflict sets |
| **FC** | O(2^(N²)) | O(2 · N³) via domain pruning | O(N²) domain snapshots |

### Graph Algorithms Used

| Algorithm | Purpose | Complexity |
|-----------|---------|------------|
| **DFS** (Depth-First Search) | Cycle detection in the slash graph | O(V + E) |
| **BFS** (Breadth-First Search) | Reachability checks for cycle validation | O(V + E) |
| **Union-Find** (conceptual) | Connected component tracking | O(α(N)) |

---

## Tech Stack & Architecture

```
┌────────────────────────────┐     HTTP/JSON      ┌──────────────────────┐
│       Frontend (Browser)   │ ◄──────────────► │    Backend (Flask)    │
│                            │    REST API        │                      │
│  • HTML5 / CSS3 / JS       │                    │  • Python 3.x       │
│  • Canvas API (charts)     │                    │  • Flask + CORS      │
│  • Web Audio API (sounds)  │                    │  • Game Logic        │
│  • Responsive UI           │                    │  • 6 Solver Engines  │
│  • Visualization Engine    │                    │  • CPU AI Module     │
└────────────────────────────┘                    └──────────────────────┘
```

**Backend Modules:**

| Module | Purpose |
|--------|---------|
| `app.py` | Flask API server with all routes |
| `game_logic.py` | Core game state, graph representation, cycle detection, scoring |
| `cpu_ai.py` | CPU AI with 5 strategies + Quicksort |
| `dp_solver.py` | DP solver (Profile DP + DP-Enhanced Backtracking) |
| `dnc_solver.py` | D&C solver, Hybrid solver, D&C cycle detection |
| `mrv_solver.py` | MRV Backtracking solver + recorded variant |
| `cbj_solver.py` | Conflict-Directed Backjumping solver + recorded variant |
| `fc_solver.py` | Forward Checking solver + recorded variant |

---

## Project Structure

```
Slant-Game-main/
├── backend/
│   ├── app.py                  # Flask REST API server (all routes)
│   ├── game_logic.py           # Core game logic, graph algorithms, scoring
│   ├── cpu_ai.py               # CPU AI strategies + Quicksort
│   ├── dp_solver.py            # Dynamic Programming solver
│   ├── dnc_solver.py           # Divide & Conquer + Hybrid solver
│   ├── mrv_solver.py           # MRV Backtracking solver
│   ├── cbj_solver.py           # Conflict-Directed Backjumping solver
│   ├── fc_solver.py            # Forward Checking solver
│   ├── cut_solver.py           # Minimum-Boundary Cut solver
│   ├── hybrid_solver.py        # Additional Hybrid utilities
│   └── test_algorithms.py      # Algorithm test suite
├── frontend/
│   ├── index.html              # Main game UI
│   ├── script.js               # Frontend logic, visualization, charts
│   └── style.css               # Styling & animations
├── README.md                   # This file
├── GAME_RULES.md               # Detailed game rules & scoring
└── strategies.txt              # In-depth algorithm documentation
```

---

## Installation & Setup

### Prerequisites
- **Python 3.7+**
- **pip** (Python package manager)
- Modern web browser (Chrome, Firefox, Edge)

### Steps

**1. Install Dependencies**
```bash
pip install flask flask-cors
```

**2. Start the Backend Server**
```bash
cd backend
python app.py
```
The Flask API starts at `http://localhost:5000`.

**3. Start the Frontend**

Open a new terminal:
```bash
python -m http.server 8000 --directory frontend
```

**4. Open in Browser**

Navigate to **http://localhost:8000**

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/state` | GET | Get current game state |
| `/api/new_game` | POST | Start a new puzzle |
| `/api/move` | POST | Make a move (human) |
| `/api/cpu_move` | POST | Trigger CPU move |
| `/api/undo` | POST | Undo last move |
| `/api/solve` | POST | Solve with selected algorithm |
| `/api/solve_dp` | POST | Solve with Dynamic Programming |
| `/api/solve_dnc` | POST | Solve with Divide & Conquer |
| `/api/solve_hybrid` | POST | Solve with Hybrid DP + D&C |
| `/api/solve_mrv` | POST | Solve with MRV Backtracking |
| `/api/solve_cbj` | POST | Solve with CBJ |
| `/api/solve_fc` | POST | Solve with Forward Checking |
| `/api/visualize` | POST | Run solver with step recording |
| `/api/compare_solvers` | POST | Benchmark all solvers |
| `/api/review` | POST | Review move correctness |
| `/api/backtrack_fix` | POST | Fix invalid board via backtracking |
| `/api/set_solver` | POST | Set CPU solver for multiplayer |

---

**Built as a DAA (Design and Analysis of Algorithms) project demonstrating practical applications of DP, Divide & Conquer, Backtracking, and Constraint Satisfaction through interactive gameplay.**
