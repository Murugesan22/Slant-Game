# SLANT - Interactive Puzzle Game

A graph theory-based puzzle game featuring greedy algorithms, constraint satisfaction, and cycle detection.

## Table of Contents
- [Overview](#overview)
- [Game Rules](#game-rules)
- [How to Play](#how-to-play)
- [Scoring System](#scoring-system)
- [Game Modes](#game-modes)
- [Installation & Setup](#installation--setup)

---

## Overview

SLANT is an educational puzzle game that demonstrates graph theory concepts through engaging gameplay. Players fill a grid with diagonal slashes while satisfying numbered constraints and avoiding loops. The game features both single-player and competitive multiplayer modes with AI opponents using three different greedy algorithm strategies.

### Key Features
- **Puzzle Solving**: Complete grids by placing diagonal slashes
- **AI Opponent**: Compete against CPU using greedy algorithms
- **Graph Theory**: Learn cycle detection and constraint satisfaction
- **Multiple Strategies**: Choose from 3 different AI behaviors
- **Responsive Design**: Beautiful UI with animations and sound effects
- **Variable Difficulty**: Multiple board sizes (3×3, 5×5, 7×7, 9×9)

---

## Game Rules

### Objective
Complete the grid by filling every cell with diagonal slashes (\ or /) while satisfying all numbered constraints and avoiding loops.

### Three Core Rules

**1. Fill Every Cell**
- Every square cell must contain exactly one diagonal slash
- Two types available:
  - **Left Slash (\)**: Connects top-left corner to bottom-right corner
  - **Right Slash (/)**: Connects top-right corner to bottom-left corner

**2. Satisfy Number Constraints**
- Some intersection points (corners where cells meet) have numbers on them
- These numbers indicate how many lines must touch that specific point
- Valid constraint numbers: 0, 1, 2, 3, or 4
- Example: A corner marked "2" must have exactly 2 lines connecting to it

**3. No Loops Allowed**
- The slashes must NOT form any closed loops or cycles
- Lines can branch and connect, but cannot circle back to form a complete loop
- Valid: Lines that branch out like a tree
- Invalid: Lines that connect back to themselves forming a cycle

---

## How to Play

### Single Player Mode (Default)

**Place Slashes**
- Single Click: Toggle between \ and / slash types
- Double Click: Clear the cell

**Visual Feedback**
- Green Numbers: Constraint satisfied
- Red Numbers: Constraint violated
- Yellow Numbers: Constraint not yet satisfied
- Red Slashes: Part of a detected loop

**Win Condition**
- All cells filled
- All constraints satisfied
- No loops present

### Multiplayer Mode

**Enable Multiplayer**
- Click the "Multiplayer" button
- Select one of three CPU strategies:
  - Strategy 1: Constraint-Focused (logical, methodical)
  - Strategy 2: Edge-First (perimeter-focused)
  - Strategy 3: Random-Greedy (unpredictable)
- Click "Confirm"

**Turn-Based Play**
- Players alternate turns (Human vs CPU)
- CPU automatically plays after each human move (1.5s delay)
- Both players compete for the highest score

**Victory**
- Player with the highest total score wins
- If scores are equal: Draw

### Controls

| Button | Function |
|--------|----------|
| New Game | Start a fresh puzzle with randomly generated constraints |
| Undo | Revert the last move |
| Multiplayer | Toggle between single-player and multiplayer modes |
| Solve | Auto-complete the puzzle using backtracking algorithm |
| Board Size | Choose grid size: 3×3, 5×5, 7×7, or 9×9 |

---

## Scoring System

Points are awarded based on strategic placement and puzzle-solving skill. Both human and CPU use identical scoring criteria for fairness.

### Scoring Criteria

**Base Move Points**
- Award: +1 point
- Condition: Every valid move
- Purpose: Rewards participation and progress

**Constraint Satisfaction Bonus**
- Award: +2 points per constraint satisfied
- Condition: When your move completes a numbered node's requirement
- Example: If a node marked "2" needs 2 connections and your move provides the 2nd connection, you earn +2 bonus points
- Maximum per move: Up to +8 points (if satisfying 4 constraints simultaneously)

**Perfect Cell Bonus**
- Award: +3 points
- Condition: All 4 corners around your cell have their constraints fully satisfied
- Purpose: Rewards creating complete, valid regions
- Note: Only applies if at least one corner has a numbered constraint

**Strategic Position Bonus**
- Award: +1 point
- Condition: Move is placed within 1 cell of the board's center
- Purpose: Encourages strategic positioning

### Scoring Examples

| Scenario | Points Breakdown | Total |
|----------|------------------|-------|
| Basic move (no constraints) | 1 (base) | 1 |
| Complete one constraint | 1 (base) + 2 (constraint) | 3 |
| Complete two constraints | 1 (base) + 2 + 2 | 5 |
| Perfect cell (all 4 corners satisfied) | 1 (base) + 4 (constraints) + 3 (perfect) | 8 |
| Optimal strategic move (center + perfect) | 1 (base) + 4 + 3 (perfect) + 1 (center) | 9 |

---

## Game Modes

### Single Player
- Description: Free-play puzzle-solving mode
- Turns: Human only
- Scoring: Player score tracked
- Goal: Complete the puzzle with valid solution
- Best For: Learning, practicing, relaxed gameplay

### Multiplayer
- Description: Competitive mode against CPU
- Turns: Alternating (Human → CPU → Human...)
- Scoring: Both players compete for highest score
- Goal: Win by outscoring the CPU
- Best For: Challenge, strategic gameplay, testing algorithms

### Solve Mode
- Description: Auto-complete using backtracking algorithm
- Turns: Algorithm takes over
- Scoring: No points awarded
- Goal: Demonstrate a valid solution exists
- Best For: Getting unstuck, learning correct patterns

---

## Installation & Setup

### Prerequisites
- Python 3.x (3.7 or higher recommended)
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Edge, Safari)

### Installation Steps

**1. Clone or Download the Project**
```bash
cd "e:\Games\backup2\DAA PROJECT"
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the Backend Server**

Open a terminal and run:
```bash
python backend/app.py
```
This starts the Flask API server on http://localhost:5000. Keep this terminal running.

**4. Start the Frontend Server**

Open a new terminal and run:
```bash
python -m http.server 8000 --directory frontend
```
This serves the UI on http://localhost:8000. Keep this terminal running too.

**5. Open in Browser**

Navigate to http://localhost:8000

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
├── README.md               # This file
├── GAME_RULES.md           # Detailed game rules
├── GREEDY_STRATEGIES.md    # AI strategy documentation
└── requirements.txt        # Python dependencies
```

---

**Enjoy playing SLANT!**
