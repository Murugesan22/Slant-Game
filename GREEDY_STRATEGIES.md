# Greedy Strategies for Slant Game

This document explains the three different greedy strategies available for the CPU opponent in multiplayer mode.

## Overview

All three strategies are **pure greedy algorithms** with no backtracking. They differ in their heuristics for selecting moves, providing variety in gameplay and CPU behavior.

---

## Strategy 1: Constraint-Focused

**Philosophy:** Prioritize cells that are adjacent to constraint nodes.

### How it works:
1. **Priority Sorting:** Cells are sorted by the number of adjacent constraint nodes
2. **Constraint Satisfaction:** Moves that satisfy constraints get higher scores
3. **Center Preference:** Small bias toward center cells for board completion
4. **Line Continuity:** Slight bonus for connecting to existing slashes

### Scoring Factors:
- ✅ **+0.5** for satisfying a constraint exactly
- ✅ **+0.2** for moving toward a constraint
- ✅ **+0.1** for connecting to existing lines
- ✅ **-0.05 × distance** from center
- ❌ **-100** for invalid moves

### Best For:
- Methodical gameplay
- Constraint-heavy boards
- Players who want a logical opponent

---

## Strategy 2: Edge-First

**Philosophy:** Start from the edges and work inward.

### How it works:
1. **Priority Sorting:** Cells are sorted by distance from center (edges first)
2. **Perimeter Focus:** Emphasizes completing the outer edges
3. **Inward Progression:** Gradually moves toward the center
4. **Constraint Awareness:** Still respects constraints but with edge priority

### Scoring Factors:
- ✅ **+0.6** for satisfying a constraint exactly
- ✅ **+0.3** for moving toward a constraint
- ✅ **+0.15 × distance** from center (edge bonus)
- ✅ **+0.05** for connecting to existing lines
- ❌ **-100** for invalid moves

### Best For:
- Different gameplay patterns
- Creating border structures first
- Players who want spatial variety

---

## Strategy 3: Random-Greedy

**Philosophy:** Add unpredictability while still being greedy.

### How it works:
1. **Valid Move Collection:** Gathers all valid moves
2. **Basic Scoring:** Uses simpler evaluation than other strategies
3. **Random Noise:** Adds random variation (-0.3 to +0.3) to scores
4. **Top Selection:** Randomly picks from the top 30% of scored moves

### Scoring Factors:
- ✅ **+0.4** for satisfying a constraint exactly
- ✅ **+0.1** for moving toward a constraint
- ✅ **+0.08** for connecting to existing lines
- 🎲 **Random noise** for unpredictability
- ❌ **-100** for invalid moves

### Best For:
- Unpredictable opponents
- Varied gameplay experiences
- Players who want surprising CPU moves

---

## How to Select a Strategy

1. Click the **"Multiplayer"** button
2. A popup will appear with the three strategy options
3. Select your preferred strategy by clicking on it
4. Click **"Confirm"** to enable multiplayer mode with that strategy
5. The CPU will use the selected strategy for all its moves

## Strategy Comparison

| Feature | Strategy 1 | Strategy 2 | Strategy 3 |
|---------|-----------|-----------|-----------|
| **Name** | Constraint-Focused | Edge-First | Random-Greedy |
| **Predictability** | High | Medium | Low |
| **Constraint Priority** | Very High | Medium | Medium |
| **Edge Preference** | Low | Very High | None |
| **Randomness** | None | None | High |
| **Best For** | Logical play | Spatial patterns | Variety |

---

## Implementation Notes

### Pure Greedy Approach
All strategies follow these principles:
- ✅ **No Backtracking:** Once a move is made, it's never reconsidered
- ✅ **Greedy Selection:** Always picks the best move according to current heuristics
- ✅ **No Look-Ahead:** Doesn't simulate future moves
- ✅ **Local Optimization:** Makes the best decision based on current state only

### Validity Checks
All strategies reject invalid moves:
- Moves that exceed constraint limits
- Moves that create more than 4 connections at a node
- Cells that are already filled

### Fair Scoring
The scoring systems are designed to be balanced:
- Human and CPU moves use the same evaluation
- No unfair advantages built into the algorithm
- Scores are based purely on greedy heuristics
