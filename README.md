# SLANT - Interactive Puzzle Game

> A graph theory-based puzzle game featuring greedy algorithms, constraint satisfaction, and cycle detection

## 📋 Table of Contents
- [Overview](#-overview)
- [Game Rules](#-game-rules)
- [How to Play](#-how-to-play)
- [Scoring System](#-scoring-system)
- [Game Modes](#-game-modes)
- [Installation & Setup](#-installation--setup)
- [Technical Details](#-technical-details)
- [Strategy Guide](#-strategy-guide)
- [Educational Value](#-educational-value)
- [Troubleshooting](#-troubleshooting)
- [License & Credits](#-license--credits)

---

## 🎮 Overview

**SLANT** is an educational puzzle game that demonstrates graph theory concepts through engaging gameplay. Players fill a grid with diagonal slashes while satisfying numbered constraints and avoiding loops. The game features both single-player and competitive multiplayer modes with AI opponents using three different greedy algorithm strategies.

### Key Features
- 🧩 **Puzzle Solving**: Complete grids by placing diagonal slashes
- 🤖 **AI Opponent**: Compete against CPU using greedy algorithms
- 📊 **Graph Theory**: Learn cycle detection and constraint satisfaction
- 🎯 **Multiple Strategies**: Choose from 3 different AI behaviors
- 📱 **Responsive Design**: Beautiful UI with animations and sound effects
- 🎲 **Variable Difficulty**: Multiple board sizes (3×3, 5×5, 7×7, 9×9)

---

## 📜 Game Rules

### Objective
Complete the grid by filling every cell with diagonal slashes (`\` or `/`) while satisfying all numbered constraints and avoiding loops.

### Three Core Rules

#### 1️⃣ Fill Every Cell
- Every square cell must contain exactly one diagonal slash
- Two types available:
  - **Left Slash (`\`)**: Connects top-left corner to bottom-right corner
  - **Right Slash (`/`)**: Connects top-right corner to bottom-left corner

#### 2️⃣ Satisfy Number Constraints
- Some intersection points (corners where cells meet) have numbers on them
- These numbers indicate how many lines must touch that specific point
- Valid constraint numbers: **0, 1, 2, 3, or 4**
- **Example**: A corner marked "2" must have exactly 2 lines connecting to it

#### 3️⃣ No Loops Allowed
- The slashes must NOT form any closed loops or cycles
- Lines can branch and connect, but cannot circle back to form a complete loop
- **Valid**: Lines that branch out like a tree
- **Invalid**: Lines that connect back to themselves forming a cycle

---

## 🎯 How to Play

### Single Player Mode (Default)

1. **Place Slashes**:
   - **Single Click**: Toggle between `\` and `/` slash types
   - **Double Click**: Clear the cell

2. **Visual Feedback**:
   - **Green Numbers**: Constraint satisfied ✅
   - **Red Numbers**: Constraint violated ❌
   - **Yellow Numbers**: Constraint not yet satisfied ⚠️
   - **Red Slashes**: Part of a detected loop 🔄

3. **Win Condition**:
   - All cells filled
   - All constraints satisfied
   - No loops present

### Multiplayer Mode

1. **Enable Multiplayer**:
   - Click the **"Multiplayer"** button
   - Select one of three CPU strategies:
     - **Strategy 1**: Constraint-Focused (logical, methodical)
     - **Strategy 2**: Edge-First (perimeter-focused)
     - **Strategy 3**: Random-Greedy (unpredictable)
   - Click **"Confirm"**

2. **Turn-Based Play**:
   - Players alternate turns (Human vs CPU)
   - CPU automatically plays after each human move (1.5s delay)
   - Both players compete for the highest score

3. **Victory**:
   - Player with the **highest total score** wins
   - If scores are equal: **Draw**

### Controls

| Button | Function |
|--------|----------|
| **New Game** | Start a fresh puzzle with randomly generated constraints |
| **Undo** | Revert the last move |
| **Multiplayer** | Toggle between single-player and multiplayer modes |
| **Solve** | Auto-complete the puzzle using backtracking algorithm |
| **Board Size** | Choose grid size: 3×3, 5×5, 7×7, or 9×9 |

---

## 💯 Scoring System

### Overview
Points are awarded based on strategic placement and puzzle-solving skill. Both human and CPU use **identical scoring criteria** for fairness.

### Scoring Criteria

#### 🎯 Base Move Points
- **Award**: +1 point
- **Condition**: Every valid move
- **Purpose**: Rewards participation and progress

#### ⭐ Constraint Satisfaction Bonus
- **Award**: +2 points per constraint satisfied
- **Condition**: When your move completes a numbered node's requirement
- **Example**: If a node marked "2" needs 2 connections and your move provides the 2nd connection, you earn +2 bonus points
- **Maximum per move**: Up to +8 points (if satisfying 4 constraints simultaneously)

#### 🏆 Perfect Cell Bonus
- **Award**: +3 points
- **Condition**: All 4 corners around your cell have their constraints fully satisfied
- **Purpose**: Rewards creating complete, valid regions
- **Note**: Only applies if at least one corner has a numbered constraint

#### 🎲 Strategic Position Bonus
- **Award**: +1 point
- **Condition**: Move is placed within 1 cell of the board's center
- **Purpose**: Encourages strategic positioning

### Scoring Examples

| Scenario | Points Breakdown | Total |
|----------|------------------|-------|
| Basic move (no constraints) | 1 (base) | **1** |
| Complete one constraint | 1 (base) + 2 (constraint) | **3** |
| Complete two constraints | 1 (base) + 2 + 2 | **5** |
| Perfect cell (all 4 corners satisfied) | 1 (base) + 4 (constraints) + 3 (perfect) | **8** |
| Optimal strategic move (center + perfect) | 1 (base) + 4 + 3 (perfect) + 1 (center) | **9** |

---

## 🎮 Game Modes

### Single Player
- **Description**: Free-play puzzle-solving mode
- **Turns**: Human only
- **Scoring**: Player score tracked
- **Goal**: Complete the puzzle with valid solution
- **Best For**: Learning, practicing, relaxed gameplay

### Multiplayer
- **Description**: Competitive mode against CPU
- **Turns**: Alternating (Human → CPU → Human...)
- **Scoring**: Both players compete for highest score
- **Goal**: Win by outscoring the CPU
- **Best For**: Challenge, strategic gameplay, testing algorithms

### Solve Mode
- **Description**: Auto-complete using backtracking algorithm
- **Turns**: Algorithm takes over
- **Scoring**: No points awarded
- **Goal**: Demonstrate a valid solution exists
- **Best For**: Getting unstuck, learning correct patterns

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.x** (3.7 or higher recommended)
- **pip** (Python package manager)
- Modern web browser (Chrome, Firefox, Edge, Safari)

### Installation Steps

1. **Clone or Download the Project**
   ```bash
   cd "e:\Games\backup2\DAA PROJECT"
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Backend Server**
   Open a terminal and run:
   ```bash
   python backend/app.py
   ```
   - This starts the Flask API server on `http://localhost:5000`
   - Keep this terminal running

4. **Start the Frontend Server**
   Open a **new terminal** and run:
   ```bash
   python -m http.server 8000 --directory frontend
   ```
   - This serves the UI on `http://localhost:8000`
   - Keep this terminal running too

5. **Open in Browser**
   Navigate to: [http://localhost:8000](http://localhost:8000)

### Project Structure
```
Slant-Game/
├── backend/
│   ├── app.py              # Flask API server
│   ├── game_logic.py       # Core game logic and graph algorithms
│   └── cpu_ai.py           # AI strategies
├── frontend/
│   ├── index.html          # Main UI
│   ├── style.css           # Styling
│   └── script.js           # Frontend logic
├── docs/
│   └── README.md           # This file
├── GAME_RULES.md           # Detailed game rules
├── GREEDY_STRATEGIES.md    # AI strategy documentation
└── requirements.txt        # Python dependencies
```

---

## 🔧 Technical Details

### Algorithms Used

#### Cycle Detection
- **Algorithm**: Depth-First Search (DFS)
- **Purpose**: Detect loops in the slash grid
- **Implementation**: Graph traversal with visited tracking
- **Time Complexity**: O(V + E) where V = nodes, E = edges

#### Constraint Satisfaction
- **Algorithm**: Degree counting at graph nodes
- **Purpose**: Verify numbered constraints are met
- **Implementation**: Count edges touching each node
- **Validation**: Real-time validation on each move

#### Greedy AI Strategies
- **Strategy 1 - Constraint-Focused**:
  - Prioritizes cells adjacent to constraint nodes
  - Scoring: +0.5 for satisfying constraints, +0.2 for approaching constraints
  
- **Strategy 2 - Edge-First**:
  - Starts from edges and works inward
  - Scoring: +0.6 for constraints, +0.15 × distance from center
  
- **Strategy 3 - Random-Greedy**:
  - Adds randomness while still being greedy
  - Scoring: +0.4 for constraints + random noise (-0.3 to +0.3)

#### Backtracking Solver
- **Algorithm**: Recursive backtracking with pruning
- **Purpose**: Auto-solve puzzles
- **Features**: Constraint validation, cycle avoidance
- **Guarantee**: Finds solution if one exists

### Technology Stack
- **Backend**: Python 3.x, Flask
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Algorithms**: Graph theory (DFS, constraint satisfaction)
- **Data Structures**: Adjacency lists, grid representation

---

## 💡 Strategy Guide

### High-Scoring Strategies

1. **Prioritize Constraint Nodes**
   - Moves that satisfy numbered nodes earn +2 bonus points each
   - Look for cells that can satisfy multiple constraints at once

2. **Create Perfect Cells**
   - Complete regions with all 4 satisfied corners for +3 bonus
   - Plan moves to create perfect cells strategically

3. **Center Positioning**
   - Fill center areas early for +1 strategic bonus
   - Center cells often affect more constraints

4. **Plan Ahead**
   - Think 2-3 moves ahead to avoid painting yourself into a corner
   - Watch out for potential loops

### Common Mistakes to Avoid

- ❌ **Ignoring constraints**: Results in low scores and invalid boards
- ❌ **Creating loops**: Causes cycle detection warnings
- ❌ **Random placement**: Earns only base points (1 per move)
- ❌ **Rushing edges**: Missing strategic center bonuses
- ❌ **Not planning**: Getting stuck with no valid moves

### Tips for Beating the CPU

- 🎯 **Against Strategy 1**: Play unpredictably, don't follow obvious constraint paths
- 🎯 **Against Strategy 2**: Claim center cells early before CPU reaches them
- 🎯 **Against Strategy 3**: Be consistent and methodical to outscore random play

---

## 🎓 Educational Value

This game demonstrates key computer science and mathematics concepts:

- **Graph Theory**: Nodes, edges, cycles, trees
- **Algorithm Design**: Greedy vs. backtracking approaches
- **Constraint Satisfaction**: NP-complete problem solving
- **Heuristic Optimization**: Trade-offs between optimality and efficiency
- **Data Structures**: Grid representations, adjacency lists
- **Game AI**: Decision-making algorithms

---

## 📞 Troubleshooting

### Backend won't start
- Ensure Python 3.x is installed: `python --version`
- Install dependencies: `pip install -r requirements.txt`
- Check if port 5000 is available

### Frontend won't load
- Ensure backend is running first
- Check if port 8000 is available
- Try a different browser
- Clear browser cache

### Game doesn't respond
- Check browser console for errors (F12)
- Verify both servers are running
- Refresh the page
- Check API connection at `http://localhost:5000/api`

---

## 📝 License & Credits

**Project**: DAA (Design and Analysis of Algorithms) Project  
**Game**: SLANT - Interactive Puzzle Game  
**Algorithms**: Graph Theory, Greedy Algorithms, Backtracking

---

**Enjoy playing SLANT! 🎮**
