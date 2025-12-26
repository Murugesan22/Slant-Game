# SLANT - Game Rules and Scoring System

## Game Objective

Complete the grid by filling every cell with diagonal slashes (\ or /) while satisfying all numbered constraints and avoiding loops.

---

## Game Rules

### Rule 1: Fill Every Cell
- Every square cell must contain exactly one diagonal slash
- Two types of slashes available:
  - **Left Slash (\\)**: Connects top-left corner to bottom-right corner
  - **Right Slash (/)**: Connects top-right corner to bottom-left corner

### Rule 2: Satisfy Number Constraints
- Some intersection points (corners where cells meet) have numbers on them
- These numbers indicate how many lines must touch that specific point
- Valid constraint numbers: 0, 1, 2, 3, or 4
- **Example**: A corner marked "2" must have exactly 2 lines connecting to it

### Rule 3: No Loops Allowed
- The slashes must NOT form any closed loops or cycles
- Lines can branch and connect, but cannot circle back to form a complete loop
- **Valid**: Lines that branch out like a tree
- **Invalid**: Lines that connect back to themselves

### Rule 4: Stay Within Bounds
- All lines must stay within the grid boundaries
- No slash can create more than 4 connections at any corner point

---

## How to Play

### Single Player Mode (Default)
1. Click any empty cell to place a left slash (\)
2. Click again to switch to right slash (/)
3. Click a third time to clear the cell
4. Continue until the board is complete

### Multiplayer Mode
1. Click the "Multiplayer: ON" button to enable turn-based play
2. Players alternate turns (Human vs CPU)
3. CPU automatically plays after each human move
4. Both players compete for the highest score

### Game Controls
- **New Game**: Start a fresh puzzle
- **Undo**: Revert the last move
- **Multiplayer**: Toggle between single-player and multiplayer modes
- **Solve**: Auto-complete the puzzle using backtracking algorithm
- **Board Size**: Choose between 3×3, 5×5, 7×7, or 9×9 grids

---

## Pointing System

### Overview
Points are awarded based on strategic placement and puzzle-solving skill. Both human and CPU players use the **exact same scoring criteria** to ensure fairness.

### Scoring Criteria

#### 🎯 Criterion 1: Base Move Points
- **Award**: +1 point
- **Condition**: Every valid move
- **Purpose**: Rewards participation and progress

#### ⭐ Criterion 2: Constraint Satisfaction Bonus
- **Award**: +2 points per constraint satisfied
- **Condition**: When your move completes a numbered node's requirement
- **Example**: If a node marked "2" needs 2 connections and your move provides the 2nd connection, you earn +2 bonus points
- **Maximum**: +4 points if one move satisfies 2 different nodes simultaneously

#### 🏆 Criterion 3: Perfect Cell Bonus
- **Award**: +3 points
- **Condition**: All 4 corners around your cell have their constraints fully satisfied
- **Purpose**: Rewards creating complete, valid regions
- **Note**: Only applies if at least one corner has a numbered constraint

#### 🎲 Criterion 4: Strategic Position Bonus
- **Award**: +1 point
- **Condition**: Move is placed within 1 cell of the board's center
- **Purpose**: Encourages strategic positioning and filling important areas

---

## Point Scoring Examples

### Example 1: Basic Move
```
Scenario: Place a slash connecting two unconstrained corners
Score: 1 point (base only)
```

### Example 2: Smart Move
```
Scenario: Your slash completes one numbered node's requirement
Score: 1 (base) + 2 (constraint) = 3 points
```

### Example 3: Excellent Move
```
Scenario: Your slash satisfies TWO numbered nodes at once
Score: 1 (base) + 2 + 2 (both constraints) = 5 points
```

### Example 4: Perfect Move
```
Scenario: Your slash creates a cell where all 4 corners are now fully satisfied
Score: 1 (base) + 4 (constraints) + 3 (perfect cell) = 8 points
```

### Example 5: Optimal Strategic Move
```
Scenario: Center cell slash that creates perfect cell and satisfies constraints
Score: 1 (base) + 4 (constraints) + 3 (perfect) + 1 (center) = 9 points
```

---

## Winning Conditions

### Game Completion
The game ends when:
1. **All cells are filled** AND
2. **All numbered constraints are satisfied** AND
3. **No loops exist**

### Determining the Winner (Multiplayer Mode)
- Player with the **highest total score** wins
- If scores are equal: **Draw**
- Victory achieved through strategic placement and constraint satisfaction

### Invalid Board State
If the board fills completely but constraints are violated or loops exist:
- Game ends with "Invalid Board State"
- No winner declared
- This demonstrates greedy algorithm limitations

---

## Strategy Tips

### High-Scoring Strategies
1. **Prioritize constraint nodes**: Moves that satisfy numbered nodes earn bonus points
2. **Create perfect cells**: Complete regions with all satisfied corners for maximum points
3. **Center positioning**: Fill center areas early for strategic bonuses
4. **Plan ahead**: Look for moves that can satisfy multiple constraints

### Common Mistakes
- ❌ Ignoring numbered constraints (results in low scores)
- ❌ Creating loops (causes invalid board state)
- ❌ Random placement without strategy (earns only base points)
- ❌ Rushing to fill edges instead of center (misses strategic bonuses)

---

## Technical Implementation

### Algorithm: Greedy Approach
- CPU uses a greedy algorithm to evaluate moves
- Each move is scored based on immediate benefit
- No backtracking during normal play (except in Solve mode)
- Both players use identical scoring logic for fairness

### Graph Theory Foundation
- Grid represented as a graph with nodes (corners) and edges (slashes)
- Cycle detection uses Depth-First Search (DFS)
- Constraint validation uses degree counting
- Pure graph algorithms ensure correctness

---

## Game Modes Summary

| Mode | Description | Turns | Scoring |
|------|-------------|-------|---------|
| **Single Player** | Free-play mode | Human only | Player score tracked |
| **Multiplayer** | Competitive mode | Alternating | Both players compete |
| **Solve** | Auto-complete | Algorithm | No scoring |

---

## Notes

- **Fairness**: Both human and CPU use identical scoring criteria
- **Skill-Based**: Better strategy and planning result in higher scores
- **Educational**: Demonstrates greedy algorithms and graph theory concepts
- **Challenging**: Requires logical thinking and spatial reasoning

---

*For technical details about implementation, see the project documentation.*
