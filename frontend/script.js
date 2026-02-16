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
let selectedSolver = 'dp'; // Track selected solver: 'dp', 'dnc', 'hybrid', 'cut'
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

// New DP and D&C solve buttons
const solveDpBtn = document.getElementById('solve-dp-btn');
const solveDncBtn = document.getElementById('solve-dnc-btn');
const solveCutBtn = document.getElementById('solve-cut-btn');
const solveHybridBtn = document.getElementById('solve-hybrid-btn');
const reviewBtn = document.getElementById('review-btn');
if (solveDpBtn) solveDpBtn.addEventListener('click', solveDP);
if (solveDncBtn) solveDncBtn.addEventListener('click', solveDnC);
if (solveCutBtn) solveCutBtn.addEventListener('click', solveBoundaryCut);
if (solveHybridBtn) solveHybridBtn.addEventListener('click', solveHybrid);
if (reviewBtn) reviewBtn.addEventListener('click', reviewMoves);

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
        'cut': 'Cut-based Partition'
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

// ... (keep rest)

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

async function solveBoundaryCut() {
    statusEl.textContent = "Solving with Minimum-Boundary Cut...";
    try {
        const response = await fetch(`${API_URL}/solve_boundary_cut`, { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            currentState = data.state;
            renderBoard(currentState);
            statusEl.textContent = data.message || "Solved with Boundary Cut!";
            playSound('cpu');
        } else {
            statusEl.textContent = data.message || "Boundary Cut Solver: No Solution Found";
            playSound('error');
        }
    } catch (e) {
        console.error(e);
        statusEl.textContent = "Boundary Cut Solver Error";
    }
}

async function solveHybrid() {
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

function getReason(data) {
    // maybe backend sends specific error?
    return "";
}

// ==============================================================================
// REVIEW FEATURE
// ==============================================================================

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
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📝</div>
                No moves to review yet!<br>
                <span style="font-size: 0.9rem; opacity: 0.7;">Make some moves to see analysis.</span>
            </div>
        `;
    } else {
        // Calculate accuracy
        const total = data.total_moves;
        const incorrect = data.incorrect_count;
        const correct = total - incorrect;
        const accuracy = total > 0 ? Math.round((correct / total) * 100) : 0;
        
        let headerColor = incorrect === 0 ? 'var(--constraint-satisfied)' : 'var(--constraint-unsatisfied)';
        let headerText = incorrect === 0 ? 'Perfect Game!' : `${incorrect} Mistake${incorrect === 1 ? '' : 's'}`;
        
        let html = `
            <div class="review-header" style="text-align: center; margin-bottom: 1.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 1rem;">
                <div style="font-size: 2.5rem; font-weight: 700; color: ${headerColor}; margin-bottom: 0.2rem;">
                    ${headerText}
                </div>
                <div style="font-size: 1rem; color: var(--text-muted);">
                    ${total} Total Moves • ${accuracy}% Accuracy
                </div>
            </div>
            
            <div class="review-list-container">
                <ul class="review-list">
        `;

        data.move_reviews.forEach((move, index) => {
            const isCorrect = move.correct;
            const statusClass = isCorrect ? 'correct' : 'incorrect';
            const statusIcon = isCorrect ? '✓' : '✗';
            
            html += `
                <li class="review-item ${statusClass}">
                    <span class="move-num">#${move.move_number}</span>
                    <span class="move-coords">${move.cell}</span>
                    <span class="move-player">${move.player === 'HUMAN' ? 'YOU' : 'CPU'}</span>
                    <span class="move-status">${statusIcon}</span>
                </li>
            `;
        });

        html += `
                </ul>
            </div>
        `;
        reviewResults.innerHTML = html;
    }

    reviewModal.classList.remove('hidden');
    // Update simple status text as fallback/supplement
    statusEl.textContent = data.incorrect_count === 0 ? "Review: Perfect!" : `Review: ${data.incorrect_count} mistakes found`;
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
