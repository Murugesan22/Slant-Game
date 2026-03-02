const API_URL = "http://127.0.0.1:5000/api";
let currentState = null;

// DOM Elements
const boardEl = document.getElementById('game-board');
const newGameBtn = document.getElementById('new-game-btn');
const undoBtn = document.getElementById('undo-btn');
const multiplayerBtn = document.getElementById('multiplayer-btn');
const solveBtn = document.getElementById('solve-btn');
const statusEl = document.getElementById('status');
const sizeBtns = document.querySelectorAll('.size-btn');

// Config
const CELL_SIZE = 60; // Must match CSS
const GRID_GAP = 2; // Must match CSS
let currentSize = 5;
let multiplayerMode = false; // Track multiplayer mode state
let selectedSolver = 'dp'; // Track selected solver: 'dp', 'dnc', 'hybrid', 'mrv', 'cbj', 'fc'
let cpuTimedOut = false; // Set when CPU exceeds time limit

// Init
document.addEventListener('DOMContentLoaded', () => {
    // Determine initial size from active button
    sizeBtns.forEach(btn => {
        if (btn.classList.contains('active')) {
            currentSize = parseInt(btn.dataset.size);
        }
        btn.addEventListener('click', (e) => {
            // Update UI
            sizeBtns.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentSize = parseInt(e.target.dataset.size);
            newGame();
        });
    });

    // FIXED: Initialize CPU score visibility (hide by default since multiplayer is off)
    const cpuScoreCard = document.querySelector('.score-card.cpu');
    const vsDivider = document.querySelector('.vs-divider');
    if (cpuScoreCard) cpuScoreCard.style.display = 'none';
    if (vsDivider) vsDivider.style.display = 'none';

    newGame();
});

newGameBtn.addEventListener('click', newGame);
undoBtn.addEventListener('click', undoLastMove);
multiplayerBtn.addEventListener('click', toggleMultiplayerMode);
solveBtn.addEventListener('click', solveGame);

// Solve Selection Modal
const solveSelectBtn = document.getElementById('solve-select-btn');
const solveSelectModal = document.getElementById('solve-select-modal');
const cancelSolveSelectBtn = document.getElementById('cancel-solve-select-btn');

// Visualize Selection Modal
const vizSelectBtn = document.getElementById('viz-select-btn');
const vizSelectModal = document.getElementById('viz-select-modal');
const cancelVizSelectBtn = document.getElementById('cancel-viz-select-btn');

// Analysis button toggles the 3 comparison charts — only after board is solved
const analysisBtn = document.getElementById('analysis-btn');
if (analysisBtn) analysisBtn.addEventListener('click', () => {
    // Guard: only allow after board has been fully solved
    const solvedStatuses = ['COMPLETED', 'WIN_HUMAN', 'WIN_CPU', 'DRAW'];
    if (!currentState || !solvedStatuses.includes(currentState.status)) {
        // Flash a warning on the status bar
        const prevText = statusEl.textContent;
        const prevColor = statusEl.style.color;
        statusEl.textContent = '⚠️ Solve the board first to view Analysis!';
        statusEl.style.color = '#fbbf24';
        playSound('error');
        setTimeout(() => {
            statusEl.textContent = prevText;
            statusEl.style.color = prevColor;
        }, 2500);
        return;
    }

    const charts = document.getElementById('compare-charts');
    if (!charts) return;
    const isHidden = charts.classList.contains('hidden');
    if (isHidden) {
        charts.classList.remove('hidden');
        loadComparisonCharts();
        analysisBtn.textContent = '📊 Analysis ✕';
        analysisBtn.style.borderColor = '#f472b6';
        analysisBtn.style.color = '#f472b6';
    } else {
        charts.classList.add('hidden');
        analysisBtn.textContent = '📊 Analysis';
        analysisBtn.style.borderColor = '#10b981';
        analysisBtn.style.color = '#10b981';
    }
    playSound('click');
});

// Review button
const reviewBtn = document.getElementById('review-btn');
if (reviewBtn) reviewBtn.addEventListener('click', reviewMoves);

// Solver dispatch map
const solverFunctions = {
    'dp': solveDP,
    'dnc': solveDnC,
    'hybrid': solveHybrid,
    'mrv': solveMRV,
    'cbj': solveCBJ,
    'fc': solveFC
};

// Open / close Solve Selection modal
if (solveSelectBtn) solveSelectBtn.addEventListener('click', () => {
    solveSelectModal.classList.remove('hidden');
});
if (cancelSolveSelectBtn) cancelSolveSelectBtn.addEventListener('click', () => {
    solveSelectModal.classList.add('hidden');
});

// Click a solver option => solve immediately & close modal
document.querySelectorAll('.solve-pick').forEach(opt => {
    opt.addEventListener('click', () => {
        const solver = opt.dataset.solver;
        solveSelectModal.classList.add('hidden');
        const fn = solverFunctions[solver];
        if (fn) fn();
    });
});

// Open / close Visualize Selection modal
if (vizSelectBtn) vizSelectBtn.addEventListener('click', () => {
    vizSelectModal.classList.remove('hidden');
});
if (cancelVizSelectBtn) cancelVizSelectBtn.addEventListener('click', () => {
    vizSelectModal.classList.add('hidden');
});

// Click a viz option => start visualization & close modal
document.querySelectorAll('.viz-pick').forEach(opt => {
    opt.addEventListener('click', () => {
        const solver = opt.dataset.solver;
        vizSelectModal.classList.add('hidden');
        startVisualization(solver);
    });
});

// Help button for instructions modal
const helpBtn = document.getElementById('help-btn');
const instructionsModal = document.getElementById('instructions-modal');
const closeInstructionsBtn = document.getElementById('close-instructions-btn');

helpBtn.addEventListener('click', () => {
    console.log("Help button clicked");
    instructionsModal.classList.remove('hidden');
    // Force visibility
    instructionsModal.style.display = 'flex';
    instructionsModal.style.opacity = '1';
    instructionsModal.style.pointerEvents = 'auto';
    playSound('click');
});

closeInstructionsBtn.addEventListener('click', () => {
    instructionsModal.classList.add('hidden');
    // Reset styles
    instructionsModal.style.display = '';
    instructionsModal.style.opacity = '';
    instructionsModal.style.pointerEvents = '';
    playSound('clear');
});

// Close modal on background click
instructionsModal.addEventListener('click', (e) => {
    if (e.target === instructionsModal) {
        instructionsModal.classList.add('hidden');
        instructionsModal.style.display = '';
        instructionsModal.style.opacity = '';
        instructionsModal.style.pointerEvents = '';
        playSound('clear');
    }
});

// Resume Audio Context on any interaction (Chrome Policy)
document.addEventListener('click', () => {
    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume().catch(e => console.log("Audio resume failed", e));
    }
}, { once: true });


async function newGame() {
    playSound('click'); // Feedback
    try {
        const response = await fetch(`${API_URL}/new_game`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ size: currentSize })
        });
        const data = await response.json();
        currentState = data;
        renderBoard(data);
        statusEl.textContent = "Player Turn";
        // Hide charts and legend from previous game
        const charts = document.getElementById('compare-charts');
        if (charts) charts.classList.add('hidden');
        const legend = document.getElementById('bt-legend');
        if (legend) legend.classList.add('hidden');
        // Reset analysis button label
        const aBtn = document.getElementById('analysis-btn');
        if (aBtn) {
            aBtn.textContent = '📊 Analysis';
            aBtn.style.borderColor = '#10b981';
            aBtn.style.color = '#10b981';
        }
    } catch (e) {
        console.error("Error starting game:", e);
        statusEl.textContent = "Error connecting to backend.";
    }
}

function renderBoard(state) {
    const size = state.size;
    const currentCells = document.querySelectorAll('.cell');

    // Check if we need to rebuild grid (Size change or first load)
    const shouldRebuild = currentCells.length !== size * size;

    if (shouldRebuild) {
        boardEl.innerHTML = '';
        // Set grid
        // We account for gap in total size: (Size * Cell) + ((Size-1) * Gap)
        const totalSize = (size * CELL_SIZE) + ((size - 1) * GRID_GAP);

        boardEl.style.gridTemplateColumns = `repeat(${size}, ${CELL_SIZE}px)`;
        boardEl.style.gap = `${GRID_GAP}px`;
        boardEl.style.width = `${totalSize}px`;
        boardEl.style.height = `${totalSize}px`; // Force height for absolute positioning of nodes

        // Create coordinate wrapper
        const coordWrapper = document.createElement('div');
        coordWrapper.classList.add('board-with-coords');

        // Create row numbers container (left side)
        const rowNumbers = document.createElement('div');
        rowNumbers.classList.add('row-numbers');
        for (let r = size; r >= 1; r--) {
            const rowLabel = document.createElement('div');
            rowLabel.classList.add('row-label');
            rowLabel.textContent = r;
            rowNumbers.appendChild(rowLabel);
        }

        // Create the actual game board container
        const boardContainer = document.createElement('div');
        boardContainer.classList.add('board-container');

        // Create the grid
        const gridEl = document.createElement('div');
        gridEl.classList.add('game-grid');
        gridEl.style.gridTemplateColumns = `repeat(${size}, ${CELL_SIZE}px)`;
        gridEl.style.gap = `${GRID_GAP}px`;
        gridEl.style.width = `${totalSize}px`;
        gridEl.style.height = `${totalSize}px`;

        // Create Cells
        for (let r = 0; r < size; r++) {
            for (let c = 0; c < size; c++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.dataset.r = r;
                cell.dataset.c = c;
                cell.addEventListener('click', () => handleCellClick(r, c));
                gridEl.appendChild(cell);
            }
        }

        boardContainer.appendChild(gridEl);

        // Create column letters container (below board)
        const colLetters = document.createElement('div');
        colLetters.classList.add('col-letters');
        for (let c = 0; c < size; c++) {
            const colLabel = document.createElement('div');
            colLabel.classList.add('col-label');
            colLabel.textContent = String.fromCharCode(65 + c); // A, B, C, ...
            colLetters.appendChild(colLabel);
        }

        // Assemble the coordinate system
        const rowAndBoard = document.createElement('div');
        rowAndBoard.classList.add('row-and-board');
        rowAndBoard.appendChild(rowNumbers);
        rowAndBoard.appendChild(boardContainer);

        coordWrapper.appendChild(rowAndBoard);
        coordWrapper.appendChild(colLetters);

        // Replace boardEl content with coordinate wrapper
        boardEl.appendChild(coordWrapper);

        // Render Constraints (Nodes) - Rebuild these too if grid changes
        // But for simplicity, we can clear constraint markers separately or rebuild all?
        // Since nodes are overlay, let's just rebuild nodes.
        // Actually, if we cleared `boardEl.innerHTML`, nodes are gone. 
        // So we need to re-add them after loop.
    }

    // Update Cells
    const cells = document.querySelectorAll('.cell'); // Re-query
    cells.forEach(cell => {
        const r = parseInt(cell.dataset.r);
        const c = parseInt(cell.dataset.c);
        const val = state.grid[r][c];

        // Manage classes without triggering re-animation if same
        const hasL = cell.classList.contains('slash-L');
        const hasR = cell.classList.contains('slash-R');

        if (val === 'L' && !hasL) {
            cell.classList.remove('slash-R');
            cell.classList.add('slash-L');
        } else if (val === 'R' && !hasR) {
            cell.classList.remove('slash-L');
            cell.classList.add('slash-R');
        } else if (val === null) {
            cell.classList.remove('slash-L', 'slash-R');
        }

        // Check if cell is in a loop and add red highlight
        cell.classList.remove('in-loop'); // Clear previous loop state
        if (state.loop_cells && state.loop_cells.length > 0) {
            const isInLoop = state.loop_cells.some(loopCell =>
                loopCell[0] === r && loopCell[1] === c
            );
            if (isInLoop) {
                cell.classList.add('in-loop');
            }
        }

        // Color CPU moves yellow in multiplayer mode
        cell.classList.remove('cpu-move');
        if (multiplayerMode && state.owners && state.owners[r][c] === 'CPU') {
            cell.classList.add('cpu-move');
        }
    });

    // Update Constraints
    // We need to manage markers carefully.
    // Simplest approach: Remove old markers, add new ones? 
    // Recreating markers is cheap compared to grid flash? 
    // Or verify if they exist?
    // Let's try to update them if they exist.

    // Constraints update with Tooltip
    // Remove old markers first to rebuild (easier logic)
    const existingMarkers = document.querySelectorAll('.constraint-marker');
    if (!shouldRebuild) existingMarkers.forEach(m => m.remove());

    const constraints = state.constraints;
    const nodeDegrees = state.node_degrees;

    // Find the grid element to append markers to
    const gridEl = document.querySelector('.game-grid') || boardEl;

    for (const key in constraints) {
        const coords = key.replace(/[()]/g, '').split(',');
        const nr = parseInt(coords[0].trim());
        const nc = parseInt(coords[1].trim());

        const limit = constraints[key];
        const currentDeg = nodeDegrees[key] || 0;

        const nodeEl = document.createElement('div');
        nodeEl.classList.add('constraint-marker');
        nodeEl.title = `Needs ${limit} lines (Current: ${currentDeg})`; // Tooltip

        if (currentDeg === limit) {
            nodeEl.classList.add('satisfied');
        } else if (currentDeg > limit) {
            nodeEl.classList.add('error');
        }

        nodeEl.textContent = limit;
        const stride = CELL_SIZE + GRID_GAP;
        const offset = GRID_GAP / 2;
        const topPos = (nr * stride) - offset;
        const leftPos = (nc * stride) - offset;

        nodeEl.style.top = `${topPos}px`;
        nodeEl.style.left = `${leftPos}px`;

        gridEl.appendChild(nodeEl);
    }

    // Status Text - SIMPLIFIED
    if (state.status === "RUNNING") {
        // Check for loop
        if (state.loop_cells && state.loop_cells.length > 0) {
            // LOOP DETECTED!
            statusEl.textContent = "Cycle Detected - Invalid Configuration";
            statusEl.style.color = "#ef4444";

            // Disable Multiplayer button when loop detected
            multiplayerBtn.disabled = true;
            multiplayerBtn.style.opacity = "0.5";
        } else {
            // Normal status
            statusEl.textContent = state.turn === 'HUMAN' ? "Player Turn" : "CPU Processing...";
            statusEl.style.color = state.turn === 'HUMAN' ? "#22c55e" : "#fbbf24";

            // Enable multiplayer button in normal state
            multiplayerBtn.disabled = false;
            multiplayerBtn.style.opacity = "1";
        }
    }

    // Update Scores
    if (state.scores) {
        const humanScoreEl = document.getElementById('score-human');
        const cpuScoreEl = document.getElementById('score-cpu');

        if (humanScoreEl) humanScoreEl.textContent = state.scores.HUMAN || 0;
        if (cpuScoreEl) cpuScoreEl.textContent = state.scores.CPU || 0;
    }

    // Check for win
    checkGameStatus(state);
}

const winOverlay = document.getElementById('win-overlay');
const closeWinBtn = document.getElementById('close-win-btn');

closeWinBtn.addEventListener('click', () => {
    winOverlay.classList.add('hidden');
});

const winContent = document.querySelector('.win-content h2');
const winMsg = document.querySelector('.win-content p');

function checkGameStatus(state) {
    // Helper function to clean up unwanted buttons
    const cleanupExtraButtons = () => {
        const winContentDiv = document.querySelector('.win-content');
        if (winContentDiv) {
            const allButtons = winContentDiv.querySelectorAll('button');
            allButtons.forEach(btn => {
                // Only keep the close button
                if (btn.id !== 'close-win-btn') {
                    btn.remove();
                }
            });
        }
    };

    // Get algorithm label based on solver
    const solverLabels = {
        'dp': 'Dynamic Programming',
        'dnc': 'Divide & Conquer',
        'hybrid': 'Hybrid (D&C + DP)',
        'mrv': 'MRV Backtracking',
        'cbj': 'Conflict-Directed Backjumping',
        'fc': 'Forward Checking'
    };
    const algoLabel = solverLabels[selectedSolver] || 'Algorithmic';

    if (state.status === "WIN_HUMAN" || state.status === "WIN_CPU" || state.status === "DRAW" || state.status === "COMPLETED") {
        if (winOverlay.classList.contains('hidden')) {
            setTimeout(() => {
                winOverlay.classList.remove('hidden');

                if (state.status === "WIN_HUMAN") {
                    winContent.textContent = "✓ Victory";
                    winMsg.textContent = `${algoLabel} algorithm completed successfully. Final Score: ${state.scores['HUMAN']} - ${state.scores['CPU']}`;
                    playSound('cpu');
                    statusEl.style.color = "#4ade80";
                } else if (state.status === "WIN_CPU") {
                    winContent.textContent = "AI Victory";
                    winMsg.textContent = `CPU ${algoLabel} algorithm outperformed player. Final Score: ${state.scores['HUMAN']} - ${state.scores['CPU']}`;
                    playSound('error');
                    statusEl.style.color = "#f43f5e";
                } else if (state.status === "DRAW") {
                    winContent.textContent = "Draw";
                    winMsg.textContent = `Both algorithms achieved equal performance. Final Score: ${state.scores['HUMAN']} - ${state.scores['CPU']}`;
                    statusEl.style.color = "#fbbf24";
                }

                cleanupExtraButtons();

            }, 100);
            statusEl.textContent = winContent.textContent;
        }
    } else if (state.status === "FILLED_INVALID") {
        statusEl.textContent = "Game Completed - Invalid Board State";
        statusEl.style.color = "#fbbf24";

        winContent.textContent = "Game Over";
        winMsg.textContent = `Board filled with constraint violations. ${algoLabel} algorithm could not find a valid solution.`;
        winOverlay.classList.remove('hidden');

        cleanupExtraButtons();

        // Add Backtrack & Fix button in multiplayer mode
        if (multiplayerMode) {
            const winContentDiv = document.querySelector('.win-content');
            if (winContentDiv && !document.getElementById('backtrack-fix-btn')) {
                const backtrackBtn = document.createElement('button');
                backtrackBtn.id = 'backtrack-fix-btn';
                backtrackBtn.className = 'btn primary backtrack-fix-btn';
                backtrackBtn.textContent = '🔄 Backtrack & Fix';
                backtrackBtn.addEventListener('click', () => {
                    winOverlay.classList.add('hidden');
                    startBacktrackFix();
                });
                // Insert before close button
                const closeBtn = document.getElementById('close-win-btn');
                winContentDiv.insertBefore(backtrackBtn, closeBtn);
            }
        }

        playSound('error');
    } else {
        if (!winOverlay.classList.contains('hidden')) {
            winOverlay.classList.add('hidden');
        }
        // ...
        // Only reset if we were previously showing win
        // Logic check: If status is RUNNING, text is handled by renderBoard
    }
}

// Sound Context
let audioCtx;
try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
} catch (e) {
    console.warn("AudioContext not supported:", e);
}

function playSound(type) {
    if (!audioCtx) return;
    if (audioCtx.state === 'suspended') {
        audioCtx.resume().catch(e => console.log("Audio resume failed", e));
    }
    const osc = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    osc.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    if (type === 'click') {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(600, audioCtx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(300, audioCtx.currentTime + 0.1);

        gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime); // Boosted volume
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.1);

        osc.start();
        osc.stop(audioCtx.currentTime + 0.1);
    } else if (type === 'clear') {
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(200, audioCtx.currentTime);
        osc.frequency.linearRampToValueAtTime(100, audioCtx.currentTime + 0.15);

        gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.15);

        osc.start();
        osc.stop(audioCtx.currentTime + 0.15);
    } else if (type === 'cpu') {
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(200, audioCtx.currentTime);
        osc.frequency.linearRampToValueAtTime(400, audioCtx.currentTime + 0.1);
        osc.frequency.linearRampToValueAtTime(300, audioCtx.currentTime + 0.2);

        gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
        gainNode.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.2);

        osc.start();
        osc.stop(audioCtx.currentTime + 0.2);
    } else if (type === 'error') {
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(150, audioCtx.currentTime);
        osc.frequency.linearRampToValueAtTime(100, audioCtx.currentTime + 0.2);

        gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.2);

        osc.start();
        osc.stop(audioCtx.currentTime + 0.2);
    }
}

// ... (previous code)

// Global timer for CPU move
let cpuMoveTimer = null;
// let isMoveCooldown = false; // Removed per user request

// Timer for detecting double-click vs single-click
let clickTimer = null;
let clickCount = 0;
let lastClickedCell = { r: -1, c: -1 };

function handleCellClickWithTimer(r, c, event) {
    clickCount++;

    // If clicking different cell, reset
    if (lastClickedCell.r !== r || lastClickedCell.c !== c) {
        clickCount = 1;
        lastClickedCell = { r, c };
    }

    if (clickTimer) {
        clearTimeout(clickTimer);
    }

    clickTimer = setTimeout(() => {
        if (clickCount === 1) {
            // Single click - Toggle
            handleCellClick(r, c);
        } else if (clickCount >= 2) {
            // Double click - Clear
            handleCellDblClick(r, c);
        }
        clickCount = 0;
        clickTimer = null;
    }, 250); // 250ms delay to detect double-click
}

async function handleCellClick(r, c) {
    if (!currentState) return;
    // if (isMoveCooldown) return; // Removed

    // Blocking Logic:
    const allowedStatuses = ["RUNNING", "FILLED_INVALID", "WIN_HUMAN", "WIN_CPU", "DRAW", "COMPLETED"];
    if (!allowedStatuses.includes(currentState.status)) return;

    // [REVIEW 1]: Strictly block interaction with CPU owned cells
    if (currentState.owners && currentState.owners[r][c] === 'CPU') {
        return;
    }

    // Block clicks while CPU is thinking in multiplayer mode (unless timed out)
    // BUT allow correction clicks on cells the human already owns (toggle L↔R)
    if (multiplayerMode && !cpuTimedOut && (cpuMoveTimer || (currentState.turn === 'CPU'))) {
        // Allow toggling human-owned cells (corrections)
        const cellOwner = currentState.owners && currentState.owners[r] && currentState.owners[r][c];
        const cellValue = currentState.grid && currentState.grid[r] && currentState.grid[r][c];
        if (cellOwner === 'HUMAN' && cellValue !== null) {
            // This is a correction — allow it, cancel pending CPU move
            if (cpuMoveTimer) {
                clearTimeout(cpuMoveTimer);
                cpuMoveTimer = null;
            }
            const wrapper = document.querySelector('.board-wrapper');
            if (wrapper) wrapper.classList.remove('cpu-thinking');
        } else {
            statusEl.textContent = "Wait - CPU is thinking...";
            return;
        }
    }
    cpuTimedOut = false; // Reset after human resumes

    // Clear any pending CPU move immediately to allow correction
    if (cpuMoveTimer) {
        clearTimeout(cpuMoveTimer);
        cpuMoveTimer = null;
    }

    const val = currentState.grid[r][c];

    // Define Cycle Preference: L <-> R (Toggle Only). CLEAR is reserved for DblClick.
    let attemptOrder = [];
    if (val === null) {
        attemptOrder = ['L', 'R']; // Try L first.
    } else if (val === 'L') {
        attemptOrder = ['R', 'L']; // Try R, fall back to L (no change if R invalid)
    } else if (val === 'R') {
        attemptOrder = ['L', 'R']; // Try L, fall back to R
    }

    for (const moveType of attemptOrder) {
        try {
            const res = await fetch(`${API_URL}/move`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ row: r, col: c, type: moveType })
            });
            const data = await res.json();

            if (data.error) {
                // If it was the last attempt, play error sound
                if (moveType === attemptOrder[attemptOrder.length - 1]) {
                    playSound('error');
                    statusEl.textContent = data.error;
                }
                continue; // Try next move in list
            }

            // Success
            currentState = data.state;
            renderBoard(currentState);

            // Check if CPU auto-fixed the board
            if (data.auto_fixed) {
                statusEl.textContent = data.message || "CPU auto-corrected its moves!";
                statusEl.style.color = "#4ade80";
                playSound('cpu');
                return;
            }

            // Check if the user needs to fix their moves
            if (data.user_fix_needed) {
                statusEl.textContent = data.message || "Please undo and change some of your moves.";
                statusEl.style.color = "#fbbf24";
                playSound('error');
                return;
            }

            playSound('click');

            // Auto CPU Trigger after small delay (Debounce) - ONLY IN MULTIPLAYER MODE
            if (multiplayerMode && currentState.status === "RUNNING" && currentState.turn === "CPU") {
                statusEl.textContent = "CPU Turn - Processing...";
                const wrapper = document.querySelector('.board-wrapper');
                if (wrapper) wrapper.classList.add('cpu-thinking');
                cpuMoveTimer = setTimeout(triggerCpuMove, 1500); // 1.5s delay
            }
            // In single-player mode, just keep the status as "Your Turn"
            else if (!multiplayerMode && currentState.status === "RUNNING") {
                statusEl.textContent = "Player Turn - Click to Place Slash";
                statusEl.style.color = "#38bdf8";
            }

            return; // Stop after successful move

        } catch (e) {
            console.error("Move failed", e);
            statusEl.textContent = `Error: ${e.message}`;
            statusEl.style.color = "#ef4444";
        }
    }
}

async function handleCellDblClick(r, c) {
    if (!currentState) return;

    // Check ownership
    if (currentState.owners && currentState.owners[r][c] === 'CPU') return;

    // Call Clear
    try {
        const res = await fetch(`${API_URL}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ row: r, col: c, type: 'CLEAR' })
        });
        const data = await res.json();

        if (data.success || !data.error) {
            currentState = data.state || data; // Handle format diffs if any
            if (data.state) currentState = data.state;

            renderBoard(currentState);
            playSound('clear');
            statusEl.textContent = "Cell Cleared";
        }
    } catch (e) {
        console.error("Clear failed", e);
    }
}

// Update Render to show Turn


// ... (rest of helper functions)

function toggleMultiplayerMode() {
    // Show solver selection modal
    const solverModal = document.getElementById('solver-modal');
    solverModal.classList.remove('hidden');

    // Set initial selection to current solver
    const solverOptions = document.querySelectorAll('.strategy-option');
    solverOptions.forEach(option => {
        option.classList.remove('selected');
        if (option.dataset.solver === selectedSolver) {
            option.classList.add('selected');
        }
    });

    playSound('click');
}

// Solver Modal Logic
document.addEventListener('DOMContentLoaded', () => {
    const solverModal = document.getElementById('solver-modal');
    const solverOptions = document.querySelectorAll('.strategy-option');
    const confirmBtn = document.getElementById('confirm-solver-btn');
    const cancelBtn = document.getElementById('cancel-solver-btn');

    // Handle solver selection
    solverOptions.forEach(option => {
        option.addEventListener('click', () => {
            // Remove selected class from all
            solverOptions.forEach(opt => opt.classList.remove('selected'));
            // Add to clicked one
            option.classList.add('selected');
            playSound('click');
        });
    });

    // Confirm button
    confirmBtn.addEventListener('click', async () => {
        const selectedOption = document.querySelector('.strategy-option.selected');
        if (selectedOption) {
            selectedSolver = selectedOption.dataset.solver;

            // Send solver to backend
            try {
                const response = await fetch(`${API_URL}/set_solver`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ solver: selectedSolver })
                });
                const data = await response.json();

                if (data.success) {
                    // Close modal
                    solverModal.classList.add('hidden');

                    // Enable multiplayer mode
                    multiplayerMode = true;

                    // Get CPU score elements
                    const cpuScoreCard = document.querySelector('.score-card.cpu');
                    const vsDivider = document.querySelector('.vs-divider');

                    // Get label
                    const label = data.label || selectedSolver.toUpperCase();

                    // Update button style
                    multiplayerBtn.textContent = `Multiplayer: ON (${label})`;
                    multiplayerBtn.style.borderColor = "#4ade80";
                    multiplayerBtn.style.color = "#4ade80";
                    statusEl.textContent = `Multiplayer Mode: ${label}`;
                    statusEl.style.color = "#4ade80";

                    // Show CPU score card
                    if (cpuScoreCard) cpuScoreCard.style.display = 'flex';
                    if (vsDivider) vsDivider.style.display = 'block';

                    // If it's CPU turn, trigger a move
                    if (currentState && currentState.status === "RUNNING" && currentState.turn === "CPU") {
                        setTimeout(triggerCpuMove, 1000);
                    }

                    playSound('cpu');
                }
            } catch (e) {
                console.error('Failed to set solver:', e);
                statusEl.textContent = 'Error setting solver';
            }
        }
    });

    // Cancel button
    cancelBtn.addEventListener('click', () => {
        solverModal.classList.add('hidden');
        playSound('clear');
    });

    // Close modal on background click
    solverModal.addEventListener('click', (e) => {
        if (e.target === solverModal) {
            solverModal.classList.add('hidden');
            playSound('clear');
        }
    });
});

// ==============================================================================
// BACKTRACKING VISUALIZATION
// ==============================================================================

let vizSteps = [];
let vizCurrentStep = 0;
let vizIsPlaying = false;
let vizTimer = null;
let vizSolver = 'mrv';
let vizFinalGrid = null;

async function startVisualization(solver) {
    vizSolver = solver;
    statusEl.textContent = `Loading ${solver.toUpperCase()} visualization...`;
    try {
        const res = await fetch(`${API_URL}/visualize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ solver })
        });
        const data = await res.json();
        if (!data.steps) { alert('Visualization failed: ' + data.message); return; }
        vizSteps = data.steps || [];
        vizFinalGrid = data.final_grid;
        vizCurrentStep = 0;
        vizIsPlaying = false;
        document.getElementById('viz-panel').classList.remove('hidden');
        document.getElementById('viz-title').textContent =
            (solver === 'mrv' ? 'MRV Backtracking' : solver === 'fc' ? 'Forward Checking' : 'Conflict-Directed Backjumping') + ' — Visualization';
        const isCBJ = solver === 'cbj';
        const isFC = solver === 'fc';
        document.getElementById('stat-jumps').classList.toggle('hidden', !isCBJ);
        document.getElementById('stat-skipped').classList.toggle('hidden', !isCBJ && !isFC);
        document.getElementById('stat-forced').classList.toggle('hidden', isCBJ);

        // Relabel "Cells Skipped" to "Pruned" for FC
        if (isFC) {
            const skippedLabel = document.querySelector('#stat-skipped .stat-label');
            if (skippedLabel) skippedLabel.textContent = 'Pruned';
        } else {
            const skippedLabel = document.querySelector('#stat-skipped .stat-label');
            if (skippedLabel) skippedLabel.textContent = 'Cells Skipped';
        }

        const stats = data.stats || {};
        document.getElementById('val-backtracks').textContent = stats.backtracks || 0;
        document.getElementById('val-jumps').textContent = stats.jumps || 0;
        document.getElementById('val-skipped').textContent = (isFC ? (stats.prunes || 0) : (stats.total_cells_skipped || 0));
        document.getElementById('val-forced').textContent = stats.forced_moves || 0;
        updateVizUI();
        clearVizHighlights();
        document.getElementById('viz-log-content').innerHTML = '';
        statusEl.textContent = data.message || 'Visualization ready';
    } catch (e) {
        console.error(e);
        statusEl.textContent = 'Visualization error';
    }
}

function updateVizUI() {
    document.getElementById('val-steps').textContent = `${vizCurrentStep} / ${vizSteps.length}`;
    const pct = vizSteps.length > 0 ? (vizCurrentStep / vizSteps.length) * 100 : 0;
    document.getElementById('viz-progress-fill').style.width = pct + '%';
}

function clearVizHighlights() {
    document.querySelectorAll('.cell').forEach(el => {
        el.classList.remove('cell-try', 'cell-forced', 'cell-backtrack', 'cell-dead-end', 'cell-jump', 'cell-conflict', 'cell-place', 'cell-skipped');
    });
}

function getCellEl(r, c) {
    return document.querySelector(`.cell[data-r="${r}"][data-c="${c}"]`);
}

function flashCell(r, c, cls, clearAfterMs) {
    const el = getCellEl(r, c);
    if (!el) return;
    el.classList.remove('cell-try', 'cell-forced', 'cell-backtrack', 'cell-dead-end', 'cell-jump', 'cell-conflict', 'cell-place', 'cell-skipped');
    el.classList.add(cls);
    if (clearAfterMs) setTimeout(() => el.classList.remove(cls), clearAfterMs);
}

function addVizLog(msg) {
    const logEl = document.getElementById('viz-log-content');
    const line = document.createElement('div');
    line.textContent = `[${vizCurrentStep}] ${msg}`;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
}

function setVizSlash(r, c, mv) {
    const el = getCellEl(r, c);
    if (!el) return;
    // Use the same CSS class mechanism as renderBoard:
    // .slash-L draws a backslash via ::before, .slash-R draws a forward slash
    el.classList.remove('slash-L', 'slash-R');
    if (mv === 'L') {
        el.classList.add('slash-L');
    } else if (mv === 'R') {
        el.classList.add('slash-R');
    }
}

function clearVizSlash(r, c) {
    setVizSlash(r, c, null);
}

function applyVizStep(step) {
    const { action, r, c, mv } = step;
    switch (action) {
        case 'MRV_SELECT':
            flashCell(r, c, 'cell-try', 0);
            addVizLog(`MRV selected (${r},${c}) — ${step.options} option(s)`);
            break;
        case 'TRY':
            flashCell(r, c, 'cell-try', 0);
            setVizSlash(r, c, mv);
            addVizLog(`Try '${mv}' at (${r},${c})`);
            break;
        case 'FORCED':
            flashCell(r, c, 'cell-forced', 0);
            setVizSlash(r, c, mv);
            addVizLog(`Forced '${mv}' at (${r},${c}) — only option`);
            break;
        case 'INVALID':
            flashCell(r, c, 'cell-dead-end', 400);
            addVizLog(`Invalid '${mv}' at (${r},${c})`);
            break;
        case 'BACKTRACK':
            flashCell(r, c, 'cell-backtrack', 0);
            clearVizSlash(r, c);
            addVizLog(`Backtrack at (${r},${c})`);
            break;
        case 'DEAD_END':
            flashCell(r, c, 'cell-dead-end', 0);
            clearVizSlash(r, c);
            addVizLog(`Dead end at (${r},${c}) — 0 options`);
            break;
        case 'PLACE':
            flashCell(r, c, 'cell-place', 0);
            setVizSlash(r, c, mv);
            addVizLog(`Placed '${mv}' at (${r},${c})`);
            break;
        case 'CONFLICT':
            flashCell(r, c, 'cell-dead-end', 0);
            flashCell(step.culprit_r, step.culprit_c, 'cell-conflict', 600);
            addVizLog(`Conflict at (${r},${c}) caused by (${step.culprit_r},${step.culprit_c})`);
            break;
        case 'JUMP':
            flashCell(step.from_r, step.from_c, 'cell-dead-end', 0);
            flashCell(step.to_r, step.to_c, 'cell-jump', 0);
            (step.skipped || []).forEach(s => {
                flashCell(s.r, s.c, 'cell-skipped', 800);
                clearVizSlash(s.r, s.c);
            });
            addVizLog(`CBJ JUMP (${step.from_r},${step.from_c}) → (${step.to_r},${step.to_c}), skipped ${(step.skipped || []).length} cells`);
            break;
        case 'MERGE':
            addVizLog(`Merge conflict sets: idx ${step.merged_from} → ${step.merged_into}`);
            break;
        case 'PRUNE':
            flashCell(r, c, 'cell-conflict', 400);
            addVizLog(`Pruned '${(step.removed || []).join(',')}' from (${r},${c}) — caused by (${step.cause_r},${step.cause_c})='${step.cause_mv}'`);
            break;
        case 'SOLVED':
            clearVizHighlights();
            addVizLog('✅ Solution found! Applying to board...');
            // Sync selectedSolver so the Victory overlay shows the correct algorithm
            selectedSolver = vizSolver;
            // Call the real solver API to get the authoritative solved state
            // (vizFinalGrid alone doesn't carry node degrees / owners / etc.)
            (async () => {
                try {
                    const routeMap = { 'cbj': 'solve_cbj', 'mrv': 'solve_mrv', 'fc': 'solve_fc' };
                    const route = routeMap[vizSolver] || 'solve_mrv';
                    const resp = await fetch(`${API_URL}/${route}`, { method: 'POST' });
                    const data = await resp.json();
                    if (data.success) {
                        currentState = data.state;
                        renderBoard(currentState);
                        statusEl.textContent = data.message || `Solved with ${vizSolver.toUpperCase()}!`;
                    }
                } catch (e) { console.error('Viz SOLVED render error', e); }
            })();
            break;
    }
    vizCurrentStep++;
    updateVizUI();
}

function stepViz() {
    if (vizCurrentStep >= vizSteps.length) { stopVizPlayback(); return; }
    applyVizStep(vizSteps[vizCurrentStep]);
}

function startVizPlayback() {
    if (vizIsPlaying) return;
    vizIsPlaying = true;
    document.getElementById('viz-play-btn').classList.add('hidden');
    document.getElementById('viz-pause-btn').classList.remove('hidden');
    const speed = parseInt(document.getElementById('viz-speed-select').value);
    function tick() {
        if (!vizIsPlaying || vizCurrentStep >= vizSteps.length) { stopVizPlayback(); return; }
        applyVizStep(vizSteps[vizCurrentStep]);
        vizTimer = setTimeout(tick, speed);
    }
    tick();
}

function stopVizPlayback() {
    vizIsPlaying = false;
    clearTimeout(vizTimer);
    document.getElementById('viz-play-btn').classList.remove('hidden');
    document.getElementById('viz-pause-btn').classList.add('hidden');
}

function resetViz() {
    stopVizPlayback();
    vizCurrentStep = 0;
    clearVizHighlights();
    // Clear all slashes drawn during visualization
    for (let r = 0; r < currentSize; r++) {
        for (let c = 0; c < currentSize; c++) {
            clearVizSlash(r, c);
        }
    }
    document.getElementById('viz-log-content').innerHTML = '';
    updateVizUI();
    // Restore board to original state before visualization
    if (currentState) renderBoard(currentState);
}

// Viz modal listeners are now in the solve-pick/viz-pick section at the top
// Individual viz buttons removed — using modal selection instead
document.getElementById('viz-play-btn').addEventListener('click', startVizPlayback);
document.getElementById('viz-pause-btn').addEventListener('click', stopVizPlayback);
document.getElementById('viz-step-btn').addEventListener('click', stepViz);
document.getElementById('viz-reset-btn').addEventListener('click', resetViz);
document.getElementById('viz-close-btn').addEventListener('click', () => {
    stopVizPlayback();
    clearVizHighlights();
    document.getElementById('viz-panel').classList.add('hidden');
    // Restore board to original state before visualization
    if (currentState) renderBoard(currentState);
});


async function triggerCpuMove() {
    cpuMoveTimer = null;
    const wrapper = document.querySelector('.board-wrapper');
    const CPU_TIMEOUT = 30; // seconds
    let secondsLeft = CPU_TIMEOUT;

    // Show initial countdown
    statusEl.textContent = `CPU Thinking... (${secondsLeft}s)`;

    // Countdown interval — updates every second
    const countdownId = setInterval(() => {
        secondsLeft--;
        if (secondsLeft > 0) {
            statusEl.textContent = `CPU Thinking... (${secondsLeft}s)`;
        }
    }, 1000);

    // AbortController to cancel fetch on timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CPU_TIMEOUT * 1000);

    try {
        const response = await fetch(`${API_URL}/cpu_move`, {
            method: 'POST',
            signal: controller.signal
        });
        const data = await response.json();

        if (data.success && data.cpu_move) {
            currentState = data.state;
            renderBoard(currentState);
            // Show message if CPU auto-corrected its moves
            if (data.auto_fixed) {
                statusEl.textContent = data.message || "CPU auto-corrected its moves!";
                statusEl.style.color = "#4ade80";
            } else if (data.message && data.message.includes("changed")) {
                statusEl.textContent = data.message;
                statusEl.style.color = "#fbbf24";
            } else {
                statusEl.textContent = "Your Turn - Click to Place Slash";
                statusEl.style.color = "#22c55e";
            }
            playSound('cpu');
        } else if (data.user_fix_needed) {
            // CPU fixed its own moves but board is still invalid — human moves are the problem
            if (data.state) {
                currentState = data.state;
                renderBoard(currentState);
            }
            statusEl.textContent = data.message || "Please change some of your moves to reach a valid solution.";
            statusEl.style.color = "#fbbf24";
            playSound('error');
        } else if (data.no_solution) {
            // CPU could not fix the board at all
            if (data.state) {
                currentState = data.state;
                renderBoard(currentState);
            }
            statusEl.textContent = data.message || "No solution found";
            statusEl.style.color = "#ef4444";
            playSound('error');
        } else {
            // CPU passed or failed
            if (data.state) {
                currentState = data.state;
                renderBoard(currentState);
            }
            statusEl.textContent = data.message || "Your Turn";
            statusEl.style.color = "#22c55e";
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            // Timeout — force turn back to human
            cpuTimedOut = true;
            statusEl.textContent = "CPU timed out! Your Turn.";
            statusEl.style.color = "#f59e0b";
            // Fetch state to resync (turn may still be CPU on backend)
            try {
                const res = await fetch(`${API_URL}/state`);
                const freshState = await res.json();
                currentState = freshState;
                renderBoard(currentState);
            } catch (_) { /* ignore */ }
        } else {
            console.error(e);
            statusEl.textContent = "CPU Processing Error";
        }
    } finally {
        clearInterval(countdownId);
        clearTimeout(timeoutId);
        if (wrapper) wrapper.classList.remove('cpu-thinking');
    }
}

// ==============================================================================
// BACKTRACK & FIX (animated recovery from FILLED_INVALID)
// ==============================================================================

let btFixSteps = [];
let btFixIdx = 0;
let btFixTimer = null;
let btFixPlaying = false;

async function startBacktrackFix() {
    // Hide the Game Over overlay so the animation is visible
    const wo = document.getElementById('win-overlay');
    if (wo) wo.classList.add('hidden');

    statusEl.textContent = "Running backtracking to fix board...";
    statusEl.style.color = "#22d3ee";

    try {
        const res = await fetch(`${API_URL}/backtrack_fix`, { method: 'POST' });
        const data = await res.json();

        if (!data.success || !data.steps) {
            statusEl.textContent = data.message || "Backtrack fix failed";
            statusEl.style.color = "#ef4444";
            playSound('error');
            return;
        }

        btFixSteps = data.steps;
        btFixIdx = 0;
        const stats = data.stats || {};
        const badCells = data.bad_cells || [];

        // Only clear the BAD cells visually — keep correct moves on the board
        document.querySelectorAll('.cell').forEach(cell => {
            cell.classList.remove('in-loop', 'cell-try', 'cell-forced', 'cell-backtrack', 'cell-dead-end', 'cell-place');
        });
        for (const [r, c] of badCells) {
            const el = getCellEl(r, c);
            if (el) {
                el.classList.remove('slash-L', 'slash-R', 'cpu-move');
                el.classList.add('cell-backtrack'); // highlight cleared cells briefly
            }
        }

        statusEl.textContent = `Fixing ${badCells.length} bad cells: 0/${btFixSteps.length} steps | Backtracks: ${stats.backtracks || 0}`;

        // Show legend
        showBacktrackLegend();

        // Delay so user can see which cells were cleared, then start animation
        btFixPlaying = true;
        setTimeout(() => animateBacktrackFix(data), 1500);

    } catch (e) {
        console.error('Backtrack fix error:', e);
        statusEl.textContent = "Backtrack fix error";
        statusEl.style.color = "#ef4444";
        playSound('error');
    }
}

function animateBacktrackFix(data) {
    const speed = 400; // ms per step — slow enough to follow each move
    const stats = data.stats || {};

    function tick() {
        if (!btFixPlaying || btFixIdx >= btFixSteps.length) {
            // Animation complete — show final solved board
            btFixPlaying = false;
            document.querySelectorAll('.cell').forEach(cell => {
                cell.classList.remove('cell-try', 'cell-forced', 'cell-backtrack', 'cell-dead-end', 'cell-place');
            });

            if (data.state) {
                currentState = data.state;
            }

            statusEl.textContent = `✓ Backtracking Complete! ${btFixSteps.length} steps, ${stats.backtracks || 0} backtracks`;
            statusEl.style.color = "#4ade80";
            hideBacktrackLegend();

            // Render the solved board
            if (data.state) {
                renderBoard(currentState);
                // Force hide the win overlay so the solved board stays visible
                const wo = document.getElementById('win-overlay');
                if (wo) wo.classList.add('hidden');
            }

            playSound('cpu');
            return;
        }

        const step = btFixSteps[btFixIdx];
        const { action, r, c, mv } = step;
        const el = getCellEl(r, c);

        if (el) {
            // Clear previous highlights on this cell
            el.classList.remove('cell-try', 'cell-forced', 'cell-backtrack', 'cell-dead-end', 'cell-place');

            switch (action) {
                case 'MRV_SELECT':
                    el.classList.add('cell-try');
                    break;
                case 'TRY':
                    el.classList.add('cell-try');
                    setVizSlash(r, c, mv);
                    break;
                case 'FORCED':
                    el.classList.add('cell-forced');
                    setVizSlash(r, c, mv);
                    break;
                case 'INVALID':
                    el.classList.add('cell-dead-end');
                    setTimeout(() => el.classList.remove('cell-dead-end'), 400);
                    break;
                case 'BACKTRACK':
                    el.classList.add('cell-backtrack');
                    clearVizSlash(r, c);
                    break;
                case 'DEAD_END':
                    el.classList.add('cell-dead-end');
                    clearVizSlash(r, c);
                    break;
                case 'PLACE':
                    el.classList.add('cell-place');
                    setVizSlash(r, c, mv);
                    break;
                case 'SOLVED':
                    break;
            }
        }

        btFixIdx++;
        statusEl.textContent = `Backtracking: ${btFixIdx}/${btFixSteps.length} steps | Backtracks: ${stats.backtracks || 0}`;

        btFixTimer = setTimeout(tick, speed);
    }

    tick();
}

function showBacktrackLegend() {
    let legend = document.getElementById('bt-legend');
    if (!legend) {
        legend = document.createElement('div');
        legend.id = 'bt-legend';
        legend.className = 'bt-legend';
        legend.innerHTML = `
            <div class="bt-legend-title">🔍 Backtracking Legend</div>
            <div class="bt-legend-items">
                <div class="bt-legend-item"><span class="bt-dot" style="background:#fbbf24"></span> Trying move</div>
                <div class="bt-legend-item"><span class="bt-dot" style="background:#4ade80"></span> Forced (only option)</div>
                <div class="bt-legend-item"><span class="bt-dot" style="background:#fb923c"></span> Backtracking</div>
                <div class="bt-legend-item"><span class="bt-dot" style="background:#ef4444"></span> Dead end</div>
                <div class="bt-legend-item"><span class="bt-dot" style="background:#22d3ee"></span> Placed</div>
            </div>
        `;
        const wrapper = document.querySelector('.board-wrapper');
        if (wrapper) {
            wrapper.insertAdjacentElement('afterend', legend);
        } else {
            document.querySelector('.app-container').appendChild(legend);
        }
    }
    legend.classList.remove('hidden');
}

function hideBacktrackLegend() {
    const legend = document.getElementById('bt-legend');
    if (legend) legend.classList.add('hidden');
}

async function undoLastMove() {
    try {
        const response = await fetch(`${API_URL}/undo`, { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            currentState = data.state;
            renderBoard(currentState);
            statusEl.textContent = "Last Move Reverted";
            playSound('clear');
        } else {
            statusEl.textContent = "No Moves to Undo";
        }
    } catch (e) {
        console.error(e);
    }
}

async function solveGame() {
    statusEl.textContent = "Solving Puzzle...";
    try {
        const response = await fetch(`${API_URL}/solve`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            currentState = data.state;
            renderBoard(currentState);
            statusEl.textContent = data.message || "Puzzle Solved Successfully";
            playSound('cpu');
        } else {
            statusEl.textContent = data.message || "No Solution Found";
            playSound('error');
        }
    } catch (e) {
        console.error(e);
        statusEl.textContent = "Solution Algorithm Error";
    }
}

async function solveDP() {
    selectedSolver = 'dp';
    statusEl.textContent = "Solving with Dynamic Programming...";
    try {
        const response = await fetch(`${API_URL}/solve_dp`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            currentState = data.state;
            renderBoard(currentState);
            statusEl.textContent = data.message || "Solved with DP!";
            playSound('cpu');
        } else {
            statusEl.textContent = data.message || "DP Solver: No Solution Found";
            playSound('error');
        }
    } catch (e) {
        console.error(e);
        statusEl.textContent = "DP Solver Error";
    }
}

async function solveDnC() {
    selectedSolver = 'dnc';
    statusEl.textContent = "Solving with Divide & Conquer...";
    try {
        const response = await fetch(`${API_URL}/solve_dnc`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            currentState = data.state;
            renderBoard(currentState);
            statusEl.textContent = data.message || "Solved with D&C!";
            playSound('cpu');
        } else {
            statusEl.textContent = data.message || "D&C Solver: No Solution Found";
            playSound('error');
        }
    } catch (e) {
        console.error(e);
        statusEl.textContent = "D&C Solver Error";
    }
}



async function solveHybrid() {
    selectedSolver = 'hybrid';
    statusEl.textContent = "Solving with Hybrid DP + D&C...";
    try {
        const response = await fetch(`${API_URL}/solve_hybrid`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            currentState = data.state;
            renderBoard(currentState);
            statusEl.textContent = data.message || "Solved with Hybrid DP + D&C!";
            playSound('cpu');
        } else {
            statusEl.textContent = data.message || "Hybrid Solver: No Solution Found";
            playSound('error');
        }
    } catch (e) {
        console.error(e);
        statusEl.textContent = "Hybrid Solver Error";
    }
}

async function solveMRV() {
    selectedSolver = 'mrv';
    statusEl.textContent = "Solving with MRV Backtracking...";
    try {
        const response = await fetch(`${API_URL}/solve_mrv`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            currentState = data.state;
            renderBoard(currentState);
            statusEl.textContent = data.message || "Solved with MRV!";
            playSound('cpu');
        } else {
            statusEl.textContent = data.message || "MRV Solver: No Solution Found";
            playSound('error');
        }
    } catch (e) {
        console.error(e);
        statusEl.textContent = "MRV Solver Error";
    }
}

async function solveCBJ() {
    selectedSolver = 'cbj';
    statusEl.textContent = "Solving with Conflict-Directed Backjumping...";
    try {
        const response = await fetch(`${API_URL}/solve_cbj`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            currentState = data.state;
            renderBoard(currentState);
            statusEl.textContent = data.message || "Solved with CBJ!";
            playSound('cpu');
        } else {
            statusEl.textContent = data.message || "CBJ Solver: No Solution Found";
            playSound('error');
        }
    } catch (e) {
        console.error(e);
        statusEl.textContent = "CBJ Solver Error";
    }
}

async function solveFC() {
    selectedSolver = 'fc';
    statusEl.textContent = "Solving with Forward Checking...";
    try {
        const response = await fetch(`${API_URL}/solve_fc`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            currentState = data.state;
            renderBoard(currentState);
            statusEl.textContent = data.message || "Solved with FC!";
            playSound('cpu');
        } else {
            statusEl.textContent = data.message || "FC Solver: No Solution Found";
            playSound('error');
        }
    } catch (e) {
        console.error(e);
        statusEl.textContent = "FC Solver Error";
    }
}

function getReason(data) {
    // maybe backend sends specific error?
    return "";
}

async function reviewMoves() {
    try {
        statusEl.textContent = "Analyzing moves...";

        const response = await fetch(`${API_URL}/review`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (data.success) {
            displayReviewResults(data);
            highlightInvalidCells(data.move_reviews || []);
            playSound('cpu');
        } else {
            statusEl.textContent = data.message || "Review failed";
            playSound('error');
        }
    } catch (e) {
        console.error('Review error:', e);
        statusEl.textContent = "Review Error";
        playSound('error');
    }
}

function displayReviewResults(data) {
    const reviewModal = document.getElementById('review-modal');
    const reviewResults = document.getElementById('review-results');

    if (!data.move_reviews || data.move_reviews.length === 0) {
        reviewResults.innerHTML = `
            <div class="success-message">
                ✓ No moves to review yet!<br>
                Start playing to see move analysis.
            </div>
        `;
    } else if (data.incorrect_count === 0) {
        reviewResults.innerHTML = `
            <div class="success-message">
                ✓ All moves were correct!<br>
                Total moves: ${data.total_moves}
            </div>
        `;
    } else {
        let html = `<h3>Move Analysis: ${data.incorrect_count} incorrect / ${data.total_moves} total</h3><ul>`;

        data.move_reviews.forEach(move => {
            const statusIcon = move.correct ? '✔' : '✘';
            const statusClass = move.correct ? 'review-correct' : 'review-incorrect';
            const statusText = move.correct ? 'Correct' : 'Incorrect';

            html += `<li>Move ${move.move_number} – <strong>${move.cell}</strong> – ${move.player} – <span class="${statusClass}">${statusIcon} ${statusText}</span></li>`;
        });

        html += '</ul>';
        reviewResults.innerHTML = html;
    }

    reviewModal.classList.remove('hidden');
    statusEl.textContent = `Review Complete: ${data.incorrect_count} incorrect move(s)`;
}

function highlightInvalidCells(moveReviews) {
    // First, clear any existing highlights
    document.querySelectorAll('.cell-correct, .cell-incorrect').forEach(cell => {
        cell.classList.remove('cell-correct', 'cell-incorrect');
    });

    // Highlight cells based on correctness
    if (moveReviews && moveReviews.length > 0) {
        moveReviews.forEach(move => {
            const cell = findCellByChessCoordinate(move.cell);
            if (cell) {
                if (move.correct) {
                    cell.classList.add('cell-correct');
                } else {
                    cell.classList.add('cell-incorrect');
                }
            }
        });
    }
}

function findCellByChessCoordinate(chessCoord) {
    // Parse chess coordinate (e.g., "A5" -> col=0, row=0 for 5x5 board)
    // Chess style: A-E columns (left to right), 1-5 rows (bottom to top)
    const col = chessCoord.charCodeAt(0) - 'A'.charCodeAt(0);
    const rowNumber = parseInt(chessCoord.substring(1));

    // Convert chess row (bottom-to-top) to grid row (top-to-bottom)
    const row = currentState.size - rowNumber;

    // Find cell in DOM
    const cells = document.querySelectorAll('.cell');
    const index = row * currentState.size + col;

    return cells[index] || null;
}

// Review modal close handler
document.addEventListener('DOMContentLoaded', () => {
    const reviewModal = document.getElementById('review-modal');
    const closeReviewBtn = document.getElementById('close-review-btn');

    if (closeReviewBtn) {
        closeReviewBtn.addEventListener('click', () => {
            reviewModal.classList.add('hidden');
            // Clear highlights when closing
            document.querySelectorAll('.cell-correct, .cell-incorrect').forEach(cell => {
                cell.classList.remove('cell-correct', 'cell-incorrect');
            });
            playSound('clear');
        });
    }

    // Close on background click
    if (reviewModal) {
        reviewModal.addEventListener('click', (e) => {
            if (e.target === reviewModal) {
                reviewModal.classList.add('hidden');
                // Clear highlights when closing
                document.querySelectorAll('.cell-correct, .cell-incorrect').forEach(cell => {
                    cell.classList.remove('cell-correct', 'cell-incorrect');
                });
                playSound('clear');
            }
        });
    }
});

// ==============================================================================
// COMPARISON CHARTS
// ==============================================================================

const CHART_COLORS = {
    'dp': '#a78bfa',
    'dnc': '#2dd4bf',
    'hybrid': '#f43f5e',
    'mrv': '#818cf8',
    'cbj': '#fbbf24',
    'fc': '#f472b6'
};

async function loadComparisonCharts() {
    const container = document.getElementById('compare-charts');
    if (!container) return;

    try {
        const res = await fetch(`${API_URL}/compare_solvers`, { method: 'POST' });
        const data = await res.json();
        if (!data.success || !data.results) return;

        const results = data.results;

        // Time chart
        const timeData = results
            .filter(r => r.time_ms >= 0)
            .map(r => ({ label: r.label, value: r.time_ms, key: r.key }));
        renderLineChart('time-chart', timeData, 'ms');

        // Space chart
        const spaceData = results
            .filter(r => r.space > 0)
            .map(r => ({ label: r.label, value: r.space, key: r.key }));
        renderBarChart('space-chart', spaceData, 'units');

        // Complexity chart (Big-O curves)
        renderComplexityChart('complexity-chart', data.board_size);

        container.classList.remove('hidden');
    } catch (e) {
        console.error('Chart load error:', e);
    }
}

function renderBarChart(containerId, data, unit) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    const maxVal = Math.max(...data.map(d => d.value), 1);

    for (const item of data) {
        const row = document.createElement('div');
        row.className = 'chart-row';

        const label = document.createElement('div');
        label.className = 'chart-label';
        label.textContent = item.label;

        const barWrap = document.createElement('div');
        barWrap.className = 'chart-bar-wrap';

        const bar = document.createElement('div');
        bar.className = 'chart-bar';
        const pct = Math.max((item.value / maxVal) * 100, 4);
        bar.style.width = pct + '%';
        bar.style.background = CHART_COLORS[item.key] || '#60a5fa';

        const val = document.createElement('span');
        val.className = 'chart-value';
        val.textContent = unit === 'ms' ? item.value.toFixed(2) + ' ms' : item.value.toLocaleString();

        barWrap.appendChild(bar);
        barWrap.appendChild(val);
        row.appendChild(label);
        row.appendChild(barWrap);
        container.appendChild(row);
    }
}

function renderLineChart(containerId, data, unit) {
    const container = document.getElementById(containerId);
    if (!container || !data.length) return;
    container.innerHTML = '';

    const W = 480, H = 260;
    const pad = { top: 30, right: 30, bottom: 60, left: 65 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const canvas = document.createElement('canvas');
    canvas.width = W * 2;  // retina
    canvas.height = H * 2;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    ctx.scale(2, 2);

    const maxVal = Math.max(...data.map(d => d.value), 0.1) * 1.15;

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 1;
    const gridLines = 5;
    for (let i = 0; i <= gridLines; i++) {
        const y = pad.top + (plotH / gridLines) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(W - pad.right, y);
        ctx.stroke();

        // Y-axis labels
        const val = maxVal - (maxVal / gridLines) * i;
        ctx.fillStyle = '#8892b0';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        ctx.fillText(val.toFixed(1), pad.left - 8, y + 3);
    }

    // Points
    const points = data.map((d, i) => {
        const x = pad.left + (plotW / (data.length - 1 || 1)) * i;
        const y = pad.top + plotH - (d.value / maxVal) * plotH;
        return { x, y, ...d };
    });

    // Gradient fill under line
    const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    gradient.addColorStop(0, 'rgba(34, 211, 238, 0.25)');
    gradient.addColorStop(1, 'rgba(34, 211, 238, 0.02)');
    ctx.beginPath();
    ctx.moveTo(points[0].x, pad.top + plotH);
    for (const p of points) ctx.lineTo(p.x, p.y);
    ctx.lineTo(points[points.length - 1].x, pad.top + plotH);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Dots + labels
    for (const p of points) {
        const color = CHART_COLORS[p.key] || '#22d3ee';
        // Dot
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Value above dot
        ctx.fillStyle = '#e2e8f0';
        ctx.font = 'bold 10px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(p.value.toFixed(2) + ' ms', p.x, p.y - 12);

        // Solver label below x-axis
        ctx.fillStyle = color;
        ctx.font = '9px Inter, sans-serif';
        ctx.textAlign = 'center';
        // Truncate long labels
        const short = p.label.length > 12 ? p.label.slice(0, 11) + '…' : p.label;
        ctx.fillText(short, p.x, pad.top + plotH + 16);
    }

    // Y-axis title
    ctx.save();
    ctx.translate(14, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#64748b';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Time (ms)', 0, 0);
    ctx.restore();
}

function renderComplexityChart(containerId, boardSize) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    const W = 480, H = 300;
    const pad = { top: 25, right: 100, bottom: 55, left: 55 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const canvas = document.createElement('canvas');
    canvas.width = W * 2;
    canvas.height = H * 2;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    ctx.scale(2, 2);

    // Theoretical complexity functions (normalized)
    const algorithms = [
        { key: 'dp', label: 'DP: O(2\u207F\u00B7n\u00B2)', fn: n => Math.pow(2, n) * n * n },
        { key: 'dnc', label: 'DnC: O(4\u00B7n\u00B2)', fn: n => 4 * n * n },
        { key: 'hybrid', label: 'Hybrid: O(2^(n/2)\u00B7n\u00B2)', fn: n => Math.pow(2, n / 2) * n * n },
        { key: 'mrv', label: 'MRV: O(n\u00B3)', fn: n => n * n * n },
        { key: 'cbj', label: 'CBJ: O(2\u00B7n\u00B3)', fn: n => 2 * n * n * n },
        { key: 'fc', label: 'FC: O(2\u00B7n\u00B3)', fn: n => 2 * n * n * n },
    ];

    const maxN = Math.max(boardSize + 2, 10);
    const nValues = [];
    for (let n = 1; n <= maxN; n++) nValues.push(n);

    // Use log scale for Y since exponential grows much faster
    const allVals = [];
    for (const algo of algorithms) {
        for (const n of nValues) allVals.push(algo.fn(n));
    }
    const maxY = Math.max(...allVals);
    const logMax = Math.log10(maxY + 1);

    // Helper to map (n, val) to canvas coords
    const toX = n => pad.left + ((n - 1) / (maxN - 1)) * plotW;
    const toY = val => {
        const logVal = Math.log10(val + 1);
        return pad.top + plotH - (logVal / logMax) * plotH;
    };

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    const gridSteps = 5;
    for (let i = 0; i <= gridSteps; i++) {
        const y = pad.top + (plotH / gridSteps) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(W - pad.right, y);
        ctx.stroke();

        // Y-axis log labels
        const logVal = logMax - (logMax / gridSteps) * i;
        ctx.fillStyle = '#8892b0';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        ctx.fillText('10^' + logVal.toFixed(1), pad.left - 6, y + 3);
    }

    // X-axis labels
    for (const n of nValues) {
        if (n % 2 === 1 || n === maxN) {
            const x = toX(n);
            ctx.fillStyle = '#8892b0';
            ctx.font = '9px JetBrains Mono, monospace';
            ctx.textAlign = 'center';
            ctx.fillText(n.toString(), x, pad.top + plotH + 14);
        }
    }

    // Vertical marker for current board size
    const markerX = toX(boardSize);
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(markerX, pad.top);
    ctx.lineTo(markerX, pad.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#fbbf24';
    ctx.font = 'bold 9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('n=' + boardSize, markerX, pad.top + plotH + 28);

    // Draw curves
    for (const algo of algorithms) {
        const color = CHART_COLORS[algo.key] || '#60a5fa';
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';

        for (let i = 0; i < nValues.length; i++) {
            const x = toX(nValues[i]);
            const y = toY(algo.fn(nValues[i]));
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Big-O label at end of line
        const lastX = toX(nValues[nValues.length - 1]);
        const lastY = toY(algo.fn(nValues[nValues.length - 1]));
        ctx.fillStyle = color;
        ctx.font = 'bold 9px JetBrains Mono, monospace';
        ctx.textAlign = 'left';
        ctx.fillText(algo.label.split(': ')[1] || algo.label, lastX + 5, lastY + 3);
    }

    // Legend below chart
    const legendY = pad.top + plotH + 36;
    let legendX = pad.left;
    ctx.font = '9px Inter, sans-serif';
    for (const algo of algorithms) {
        const color = CHART_COLORS[algo.key] || '#60a5fa';
        // dot
        ctx.fillStyle = color;
        ctx.fillRect(legendX, legendY, 8, 8);
        // text
        ctx.fillStyle = '#cbd5e1';
        ctx.textAlign = 'left';
        ctx.fillText(algo.label, legendX + 11, legendY + 7);
        legendX += ctx.measureText(algo.label).width + 20;
    }

    // Axis titles
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Board Size (n)', pad.left + plotW / 2, pad.top + plotH + 14);

    ctx.save();
    ctx.translate(12, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Operations (log scale)', 0, 0);
    ctx.restore();
}


