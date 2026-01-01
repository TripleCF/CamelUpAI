/**
 * Camel Up Scenario Builder - Frontend Application
 */

// Constants
const CAMEL_NAMES = ['Blue', 'Green', 'Orange', 'Yellow', 'White'];
const CAMEL_EMOJIS = ['🔵', '🟢', '🟠', '🟡', '⚪'];
const NUM_SPACES = 16;

// State
let currentState = null;
let draggedCamel = null;
let autoPlayInterval = null;
let roundNumber = 0;

// DOM Elements
const gameBoard = document.getElementById('gameBoard');
const tilesList = document.getElementById('tilesList');
const diceList = document.getElementById('diceList');
const rankingList = document.getElementById('rankingList');
const modelPrediction = document.getElementById('modelPrediction');
const mcPrediction = document.getElementById('mcPrediction');
const winChart = document.getElementById('winChart');
const chartLegend = document.getElementById('chartLegend');
const rollModal = document.getElementById('rollModal');
const rollCamel = document.getElementById('rollCamel');
const rollDistance = document.getElementById('rollDistance');
const actionLog = document.getElementById('actionLog');
const simStepBtn = document.getElementById('simStepBtn');
const simAutoBtn = document.getElementById('simAutoBtn');
const simSpeed = document.getElementById('simSpeed');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeBoard();
    initializeLegend();
    setupEventListeners();
    fetchState();
});

function initializeBoard() {
    gameBoard.innerHTML = '';
    for (let i = 0; i < NUM_SPACES; i++) {
        const space = document.createElement('div');
        space.className = `board-space ${i >= 14 ? 'space-finish' : ''}`;
        space.dataset.space = i;
        space.innerHTML = `<span class="space-number">${i + 1}</span>`;

        // Drag & drop
        space.addEventListener('dragover', handleDragOver);
        space.addEventListener('dragleave', handleDragLeave);
        space.addEventListener('drop', handleDrop);

        gameBoard.appendChild(space);
    }
}



function initializeLegend() {
    chartLegend.innerHTML = `
        <div class="legend-item">
            <div class="legend-color win"></div>
            <span>1st Place</span>
        </div>
        <div class="legend-item">
            <div class="legend-color second"></div>
            <span>2nd Place</span>
        </div>
    `;
}

function createCamelToken(camelId) {
    const token = document.createElement('div');
    token.className = `camel-token camel-${CAMEL_NAMES[camelId].toLowerCase()}`;
    token.textContent = '🐪';
    token.dataset.camelId = camelId;
    token.draggable = true;

    token.addEventListener('dragstart', handleDragStart);
    token.addEventListener('dragend', handleDragEnd);

    return token;
}

function setupEventListeners() {
    document.getElementById('resetBtn').addEventListener('click', resetState);
    document.getElementById('rollBtn').addEventListener('click', rollDice);
    document.getElementById('mcSimulations').addEventListener('change', fetchMonteCarlo);
    document.getElementById('runSimsBtn').addEventListener('click', runWinSimulations);

    rollModal.addEventListener('click', () => {
        rollModal.classList.remove('show');
    });

    // Simulation mode controls
    simStepBtn.addEventListener('click', runSimulationStep);
    simAutoBtn.addEventListener('click', toggleAutoPlay);
}

// API Functions
async function fetchState() {
    try {
        const response = await fetch('/api/state');
        const state = await response.json();
        currentState = state;
        renderState(state);
        fetchPredictions();
    } catch (error) {
        console.error('Error fetching state:', error);
    }
}

async function updateState(newState) {
    try {
        const response = await fetch('/api/state', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newState),
        });
        const state = await response.json();
        currentState = state;
        renderState(state);
        fetchPredictions();
    } catch (error) {
        console.error('Error updating state:', error);
    }
}

async function resetState() {
    stopAutoPlay();
    clearLog();
    try {
        const response = await fetch('/api/reset', { method: 'POST' });
        const state = await response.json();
        currentState = state;
        renderState(state);
        fetchPredictions();
    } catch (error) {
        console.error('Error resetting state:', error);
    }
}

async function rollDice() {
    try {
        const response = await fetch('/api/roll', { method: 'POST' });
        const result = await response.json();

        if (result.error) {
            alert(result.error);
            return;
        }

        // Show roll animation
        showRollModal(result.camelId, result.distance);

        // Update state
        currentState = result.state;
        renderState(result.state);
        fetchPredictions();
    } catch (error) {
        console.error('Error rolling dice:', error);
    }
}

async function fetchPredictions() {
    fetchModelPrediction();
    fetchMonteCarlo();
}

async function fetchModelPrediction() {
    modelPrediction.innerHTML = '<div class="loading">Analyzing...</div>';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });
        const prediction = await response.json();
        renderModelPrediction(prediction);
    } catch (error) {
        modelPrediction.innerHTML = '<div class="error">Error loading prediction</div>';
    }
}

async function fetchMonteCarlo() {
    mcPrediction.innerHTML = '<div class="loading">Calculating...</div>';

    const simulations = parseInt(document.getElementById('mcSimulations').value);

    try {
        // Use full game endpoint for 18-action EVs
        const response = await fetch('/api/monte-carlo-full', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ simulations }),
        });
        const prediction = await response.json();
        renderMCPrediction(prediction);
    } catch (error) {
        mcPrediction.innerHTML = '<div class="error">Error loading MC analysis</div>';
    }
}

async function runWinSimulations() {
    winChart.innerHTML = '<div class="loading">Running simulations...</div>';

    const simulations = parseInt(document.getElementById('winSimulations').value);

    try {
        const response = await fetch('/api/simulate-wins', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ simulations }),
        });
        const result = await response.json();
        renderWinChart(result);
    } catch (error) {
        winChart.innerHTML = '<div class="error">Error running simulations</div>';
    }
}

// Render Functions
function renderState(state) {
    // Clear board camels
    document.querySelectorAll('.board-space .camel-token').forEach(el => el.remove());

    // Place camels on board
    // The server sends camels bottom-to-top, and CSS uses column-reverse
    // So we need to add them in bottom-to-top order (first camel = bottom = added first)
    state.board.forEach(space => {
        const spaceEl = document.querySelector(`.board-space[data-space="${space.space}"]`);
        // Add camels in order (first = bottom of stack, will appear at bottom due to column-reverse)
        space.camels.forEach(camel => {
            const token = createCamelToken(camel.id);
            spaceEl.appendChild(token);
        });
    });

    // Update dice remaining
    renderDice(state.diceRemaining);

    // Update rankings
    renderRankings(state.rankings);

    // Update betting tiles
    renderTiles(state.tilesRemaining);

    // Update roll button
    const rollBtn = document.getElementById('rollBtn');
    rollBtn.disabled = state.isLegComplete;
    if (state.isLegComplete) {
        rollBtn.textContent = '✅ Leg Complete';
    } else {
        rollBtn.textContent = `🎲 Roll Dice (${state.diceRemaining.length} left)`;
    }

    // Update full game state if present
    if (state.currentLeg !== undefined) {
        renderGameProgress(state);
        renderGameEndBets(state);
        renderDesertTiles(state);
    }
}

function renderTiles(tilesRemaining) {
    const ALL_TILES = [5, 3, 2, 1];

    tilesList.innerHTML = CAMEL_NAMES.map((name, id) => {
        const available = tilesRemaining[name] || [];
        const tilesHtml = ALL_TILES.map(value => {
            const isAvailable = available.includes(value);
            return `<span class="tile-chip ${isAvailable ? 'tile-available' : 'tile-taken'}">${value}</span>`;
        }).join('');

        return `
            <div class="tiles-camel-row">
                <div class="camel-token camel-${name.toLowerCase()}" style="width: 24px; height: 20px; font-size: 0.9rem;">🐪</div>
                <span style="min-width: 50px; font-size: 0.85rem;">${name}</span>
                <div class="tiles-values">${tilesHtml}</div>
            </div>
        `;
    }).join('');
}

function renderDice(diceRemaining) {
    const remainingIds = new Set(diceRemaining.map(d => d.id));

    diceList.innerHTML = CAMEL_NAMES.map((name, id) => {
        const isRemaining = remainingIds.has(id);
        return `
            <div class="dice-item ${isRemaining ? '' : 'dice-used'}">
                <div class="camel-token camel-${name.toLowerCase()}" style="width: 24px; height: 20px; font-size: 0.9rem;">🐪</div>
                ${name}
            </div>
        `;
    }).join('');
}

function renderRankings(rankings) {
    rankingList.innerHTML = rankings.map(r => `
        <div class="rank-item">
            <span class="rank-position">#${r.rank}</span>
            <div class="camel-token camel-${r.camel.toLowerCase()}" style="width: 24px; height: 20px; font-size: 0.9rem;">🐪</div>
            ${r.camel}
        </div>
    `).join('');
}

function renderGameProgress(state) {
    const legCounter = document.getElementById('legCounter');
    const leaderPosition = document.getElementById('leaderPosition');
    const gameStatus = document.getElementById('gameStatus');

    if (legCounter) {
        legCounter.innerHTML = `Leg: <span class="progress-value">${state.currentLeg || 1}</span>`;
    }
    if (leaderPosition) {
        // Show progress toward finish line (space 16)
        const leaderSpace = (state.leaderPosition || 0) + 1;
        const progressPercent = Math.round((leaderSpace / 16) * 100);
        leaderPosition.innerHTML = `Finish Progress: <span class="progress-value">${progressPercent}% (Space ${leaderSpace}/16)</span>`;
    }
    if (gameStatus) {
        const isComplete = state.gameComplete;
        const statusText = isComplete ? 'Game Complete!' : 'In Progress';
        const statusClass = isComplete ? 'status-complete' : 'status-active';
        gameStatus.innerHTML = `Status: <span class="progress-value ${statusClass}">${statusText}</span>`;
    }
}

function renderDesertTiles(state) {
    // Clear existing desert tile markers
    document.querySelectorAll('.desert-tile-marker').forEach(el => el.remove());

    if (!state.desertTiles) return;

    for (const [spaceStr, tileInfo] of Object.entries(state.desertTiles)) {
        const space = parseInt(spaceStr);
        const spaceEl = document.querySelector(`.board-space[data-space="${space}"]`);
        if (spaceEl) {
            const marker = document.createElement('div');
            marker.className = `desert-tile-marker ${tileInfo.type}`;
            marker.textContent = tileInfo.type === 'oasis' ? '🏝️' : '🌵';
            marker.title = `${tileInfo.type} (Player ${tileInfo.owner})`;
            spaceEl.appendChild(marker);
        }
    }
}

function renderGameEndBets(state) {
    const winnerBetButtons = document.getElementById('winnerBetButtons');
    const loserBetButtons = document.getElementById('loserBetButtons');

    if (!winnerBetButtons || !loserBetButtons) return;

    const rankings = state.rankings || [];
    const playerWinnerBets = (state.gameWinnerBets && state.gameWinnerBets[0]) || [];
    const playerLoserBets = (state.gameLoserBets && state.gameLoserBets[0]) || [];

    // Helper to get camel ID from name
    const getCamelId = (camelName) => CAMEL_NAMES.findIndex(n => n === camelName);

    // Render winner bet buttons
    winnerBetButtons.innerHTML = rankings.map(r => {
        const camelId = getCamelId(r.camel);
        const alreadyBet = playerWinnerBets.includes(camelId);
        const btnClass = alreadyBet ? 'bet-btn used' : 'bet-btn';
        return `
            <button class="${btnClass} camel-${r.camel.toLowerCase()}" 
                    ${alreadyBet ? 'disabled' : ''} 
                    data-action="winner" data-camel="${r.camel}">
                ${CAMEL_EMOJIS[camelId]} ${r.camel}
            </button>
        `;
    }).join('');

    // Render loser bet buttons
    loserBetButtons.innerHTML = rankings.map(r => {
        const camelId = getCamelId(r.camel);
        const alreadyBet = playerLoserBets.includes(camelId);
        const btnClass = alreadyBet ? 'bet-btn used' : 'bet-btn';
        return `
            <button class="${btnClass} camel-${r.camel.toLowerCase()}" 
                    ${alreadyBet ? 'disabled' : ''} 
                    data-action="loser" data-camel="${r.camel}">
                ${CAMEL_EMOJIS[camelId]} ${r.camel}
            </button>
        `;
    }).join('');
}

function renderModelPrediction(prediction) {
    if (prediction.error) {
        modelPrediction.innerHTML = `<div class="error">${prediction.error}</div>`;
        return;
    }

    const bestAction = prediction.action;
    const probs = prediction.actionProbabilities;
    const maxProb = Math.max(...probs.map(p => p.prob));

    // Identify action categories by name pattern
    const isGameWinner = (name) => name.startsWith('Winner');
    const isGameLoser = (name) => name.startsWith('Loser');
    const isDesert = (name) => name.includes('Oasis') || name.includes('Mirage');

    // Find best game winner/loser by probability
    const gameWinners = probs.filter(p => isGameWinner(p.name));
    const gameLosers = probs.filter(p => isGameLoser(p.name));
    const bestWinner = gameWinners.length > 0 ? gameWinners.reduce((a, b) => a.prob > b.prob ? a : b) : null;
    const bestLoser = gameLosers.length > 0 ? gameLosers.reduce((a, b) => a.prob > b.prob ? a : b) : null;

    let actionsHtml = '';

    // Show roll and leg bet actions
    for (let i = 0; i < probs.length; i++) {
        const p = probs[i];
        const isValid = prediction.validActions[i];

        // Skip game winner/loser and desert (we'll add condensed versions)
        if (isGameWinner(p.name) || isGameLoser(p.name) || isDesert(p.name)) continue;

        const isBest = p.prob === maxProb && p.prob > 0;
        actionsHtml += `
            <div class="prob-row ${isValid ? '' : 'invalid'}">
                <div class="prob-label">${p.name}</div>
                <div class="prob-bar-container">
                    <div class="prob-bar ${isBest ? 'best' : ''}" style="width: ${p.prob * 100}%"></div>
                </div>
                <div class="prob-value">${(p.prob * 100).toFixed(1)}%</div>
            </div>
        `;
    }

    // Condensed desert tile
    const desertProbs = probs.filter(p => isDesert(p.name));
    if (desertProbs.length > 0) {
        const totalDesertProb = desertProbs.reduce((s, p) => s + p.prob, 0);
        actionsHtml += `
            <div class="prob-row">
                <div class="prob-label">🏜️ Desert Tile</div>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: ${totalDesertProb * 100}%"></div>
                </div>
                <div class="prob-value">${(totalDesertProb * 100).toFixed(1)}%</div>
            </div>
        `;
    }

    // Condensed game winner (best only)
    if (bestWinner) {
        const isBest = bestWinner.prob === maxProb;
        actionsHtml += `
            <div class="prob-row">
                <div class="prob-label">🏆 ${bestWinner.name}</div>
                <div class="prob-bar-container">
                    <div class="prob-bar ${isBest ? 'best' : ''}" style="width: ${bestWinner.prob * 100}%"></div>
                </div>
                <div class="prob-value">${(bestWinner.prob * 100).toFixed(1)}%</div>
            </div>
        `;
    }

    // Condensed game loser (best only)
    if (bestLoser) {
        const isBest = bestLoser.prob === maxProb;
        actionsHtml += `
            <div class="prob-row">
                <div class="prob-label">💀 ${bestLoser.name}</div>
                <div class="prob-bar-container">
                    <div class="prob-bar ${isBest ? 'best' : ''}" style="width: ${bestLoser.prob * 100}%"></div>
                </div>
                <div class="prob-value">${(bestLoser.prob * 100).toFixed(1)}%</div>
            </div>
        `;
    }

    modelPrediction.innerHTML = `
        <div class="recommendation">
            <div class="rec-icon">${bestAction === 0 ? '🎲' : '🎟️'}</div>
            <div>
                <div class="rec-action">${prediction.actionName}</div>
                <div class="rec-subtitle">Recommended action</div>
            </div>
        </div>
        <div class="action-probs">
            ${actionsHtml}
        </div>
    `;
}

function renderMCPrediction(prediction) {
    if (prediction.error) {
        mcPrediction.innerHTML = `<div class="error">${prediction.error}</div>`;
        return;
    }

    const bestAction = prediction.action;
    const evs = prediction.expectedValues;

    // Support both old format (immediateEv/strategicEv) and new format (ev/category)
    const hasCategories = evs.length > 0 && evs[0].category !== undefined;
    const maxEv = Math.max(...evs.filter(e => e.isValid).map(e => hasCategories ? e.ev : e.strategicEv));

    // Group by category
    const categories = {};
    for (const e of evs) {
        const cat = e.category || 'leg_bet';
        if (!categories[cat]) categories[cat] = [];
        categories[cat].push(e);
    }

    // Find best valid action in game_winner and game_loser categories
    const getBest = (cat) => {
        const items = categories[cat] || [];
        const valid = items.filter(e => e.isValid);
        if (valid.length === 0) return null;
        return valid.reduce((a, b) => (hasCategories ? a.ev : a.strategicEv) > (hasCategories ? b.ev : b.strategicEv) ? a : b);
    };

    const bestWinner = getBest('game_winner');
    const bestLoser = getBest('game_loser');

    let actionsHtml = '';

    // Dice and Leg bet actions (show all)
    for (const e of evs) {
        if (e.category === 'game_winner' || e.category === 'game_loser' || e.category === 'desert') continue;
        const ev = hasCategories ? e.ev : e.strategicEv;
        const isBest = ev === maxEv && e.isValid;
        actionsHtml += `
            <div class="prob-row ${e.isValid ? '' : 'invalid'}">
                <div class="prob-label">${e.name}</div>
                <div class="ev-value ${isBest ? 'best-ev' : ''}">${e.isValid ? ev.toFixed(2) : '-'}</div>
            </div>
        `;
    }

    // Desert tiles (condensed)
    const desertActions = categories['desert'] || [];
    if (desertActions.length > 0) {
        const anyValid = desertActions.some(e => e.isValid);
        actionsHtml += `
            <div class="prob-row ${anyValid ? '' : 'invalid'}">
                <div class="prob-label">🏜️ Desert Tile</div>
                <div class="ev-value">${anyValid ? '0.50' : '-'}</div>
            </div>
        `;
    }

    // Game winner (show only best)
    if (bestWinner) {
        const ev = hasCategories ? bestWinner.ev : bestWinner.strategicEv;
        const isBest = ev === maxEv;
        actionsHtml += `
            <div class="prob-row">
                <div class="prob-label">🏆 ${bestWinner.name}</div>
                <div class="ev-value ${isBest ? 'best-ev' : ''}">${ev.toFixed(2)}</div>
            </div>
        `;
    }

    // Game loser (show only best)
    if (bestLoser) {
        const ev = hasCategories ? bestLoser.ev : bestLoser.strategicEv;
        const isBest = ev === maxEv;
        actionsHtml += `
            <div class="prob-row">
                <div class="prob-label">💀 ${bestLoser.name}</div>
                <div class="ev-value ${isBest ? 'best-ev' : ''}">${ev.toFixed(2)}</div>
            </div>
        `;
    }

    mcPrediction.innerHTML = `
        <div class="recommendation">
            <div class="rec-icon">${bestAction === 0 ? '🎲' : '🎟️'}</div>
            <div>
                <div class="rec-action">${prediction.actionName}</div>
                <div class="rec-subtitle">Best expected value</div>
            </div>
        </div>
        <div class="ev-header">
            <span class="ev-label">Action</span>
            <span class="ev-value-header">EV</span>
        </div>
        <div class="action-probs">
            ${actionsHtml}
        </div>
        <div class="ev-legend">
            <span><strong>EV:</strong> Expected coins from this action</span>
        </div>
    `;
}

function renderWinChart(result) {
    if (result.error) {
        winChart.innerHTML = `<div class="error">${result.error}</div>`;
        return;
    }

    const maxProb = Math.max(...result.results.map(r => Math.max(r.winProb, r.secondProb)));

    winChart.innerHTML = result.results.map(r => {
        const winHeight = (r.winProb / maxProb) * 100;
        const secondHeight = (r.secondProb / maxProb) * 100;

        return `
            <div class="chart-bar-group">
                <div class="chart-bars">
                    <div class="chart-bar win-bar" style="height: ${winHeight}%">
                        <span class="chart-bar-value">${(r.winProb * 100).toFixed(0)}%</span>
                    </div>
                    <div class="chart-bar second-bar" style="height: ${secondHeight}%">
                        <span class="chart-bar-value">${(r.secondProb * 100).toFixed(0)}%</span>
                    </div>
                </div>
                <div class="chart-bar-label">
                    <div class="camel-token camel-${r.camelName.toLowerCase()}" style="width: 24px; height: 18px; font-size: 0.8rem;">🐪</div>
                </div>
            </div>
        `;
    }).join('');
}

function showRollModal(camelId, distance) {
    const camelName = CAMEL_NAMES[camelId];
    rollCamel.className = `roll-camel camel-${camelName.toLowerCase()}`;
    rollCamel.textContent = `🐪 ${camelName}`;
    rollDistance.textContent = `Moved ${distance} space${distance > 1 ? 's' : ''}`;
    rollModal.classList.add('show');

    setTimeout(() => {
        rollModal.classList.remove('show');
    }, 1500);
}

// Drag & Drop Handlers
function handleDragStart(e) {
    draggedCamel = e.target;
    e.target.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
}

function handleDragEnd(e) {
    e.target.classList.remove('dragging');
    draggedCamel = null;
    // Clear all drag-over states to prevent lingering highlights
    document.querySelectorAll('.board-space.drag-over').forEach(s => s.classList.remove('drag-over'));
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    // Only add if not already present
    if (!e.currentTarget.classList.contains('drag-over')) {
        e.currentTarget.classList.add('drag-over');
    }
}

function handleDragLeave(e) {
    // Check if we're actually leaving the element (not entering a child)
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX;
    const y = e.clientY;
    if (x < rect.left || x >= rect.right || y < rect.top || y >= rect.bottom) {
        e.currentTarget.classList.remove('drag-over');
    }
}

function handleDrop(e) {
    e.preventDefault();
    // Clear all drag-over states on drop
    document.querySelectorAll('.board-space.drag-over').forEach(s => s.classList.remove('drag-over'));

    if (!draggedCamel) return;

    const targetSpace = parseInt(e.currentTarget.dataset.space);
    const camelId = parseInt(draggedCamel.dataset.camelId);

    // Build new state
    const newBoard = [];
    for (let i = 0; i < NUM_SPACES; i++) {
        const spaceEl = document.querySelector(`.board-space[data-space="${i}"]`);
        const camels = Array.from(spaceEl.querySelectorAll('.camel-token'))
            .map(t => parseInt(t.dataset.camelId))
            .filter(id => id !== camelId); // Remove the dragged camel from its current position

        newBoard.push({
            space: i,
            camels: camels.map(id => ({ id })),
        });
    }

    // Add camel to target space (on top)
    newBoard[targetSpace].camels.push({ id: camelId });

    // Update state on server
    updateState({
        board: newBoard,
        diceRemaining: currentState?.diceRemaining || [],
        tilesRemaining: currentState?.tilesRemaining || {},
        playerBets: currentState?.playerBets || [[], [], [], []],
        playerCoins: currentState?.playerCoins || [0, 0, 0, 0],
    });
}

// Simulation Mode Functions
async function runSimulationStep() {
    simStepBtn.disabled = true;
    simStepBtn.textContent = '⏳ Running...';

    try {
        const response = await fetch('/api/simulate-step', { method: 'POST' });
        const result = await response.json();

        if (result.error) {
            addLogEntry('System', result.error, 'error');
            return;
        }

        // Add round marker if this is a new leg
        if (roundNumber === 0 || (result.legComplete && result.actions.length === 0)) {
            roundNumber++;
            if (result.actions.length === 0 && result.legComplete) {
                addLogEntry('System', '🏁 Leg already complete! Click Reset to start a new game.', 'system');
                stopAutoPlay();
                return;
            }
        }

        // Log each action
        result.actions.forEach(action => {
            addLogEntry(action.playerName, action.actionName, `player-${action.player}`);
        });

        // Update state
        currentState = result.state;

        // Safety check: specific case where no actions generated
        if (result.actions.length === 0 && !result.legComplete && !result.gameComplete) {
            addLogEntry('System', '⚠️ Implementation Warning: No actions generated this step.', 'error');
            // Stop autoplay to prevent infinite loop
            stopAutoPlay();
            return;
        }

        renderState(result.state);
        fetchPredictions();

        // Check if leg complete
        if (result.legComplete) {
            const rankings = result.finalRankings || [];
            addLogEntry('System', `🏁 Leg Complete! Winner: ${rankings[0] || 'Unknown'}`, 'result');

            if (result.betResults) {
                const resultText = Object.entries(result.betResults)
                    .map(([player, delta]) => `${player}: ${delta >= 0 ? '+' : ''}${delta}`)
                    .join(', ');
                addLogEntry('System', `💰 Bet Results: ${resultText}`, 'result');
            }

            stopAutoPlay();
        }

    } catch (error) {
        console.error('Error running simulation step:', error);
        addLogEntry('System', 'Error running simulation', 'error');
    } finally {
        simStepBtn.disabled = false;
        simStepBtn.textContent = '▶ Step';
    }
}

function toggleAutoPlay() {
    if (autoPlayInterval) {
        stopAutoPlay();
    } else {
        startAutoPlay();
    }
}

function startAutoPlay() {
    const speed = parseInt(simSpeed.value);
    simAutoBtn.classList.add('auto-active');
    simAutoBtn.textContent = '⏹ Stop';

    // Run first step immediately
    runSimulationStep();

    // Set up interval for subsequent steps
    autoPlayInterval = setInterval(async () => {
        if (currentState?.isLegComplete) {
            stopAutoPlay();
            return;
        }
        await runSimulationStep();
    }, speed);
}

function stopAutoPlay() {
    if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = null;
    }
    simAutoBtn.classList.remove('auto-active');
    simAutoBtn.textContent = '⏯ Auto-play';
}

function addLogEntry(player, action, className = '') {
    // Remove placeholder if present
    const placeholder = actionLog.querySelector('.log-placeholder');
    if (placeholder) {
        placeholder.remove();
    }

    const entry = document.createElement('div');
    entry.className = `log-entry ${className}`;

    if (className === 'result' || className === 'system' || className === 'error') {
        entry.innerHTML = `<span class="log-action">${action}</span>`;
        entry.className = className === 'result' ? 'log-result' : 'log-entry';
    } else {
        entry.innerHTML = `
            <span class="log-player ${className}">${player}</span>
            <span class="log-action">${action}</span>
        `;
    }

    actionLog.appendChild(entry);
    actionLog.scrollTop = actionLog.scrollHeight;
}

function clearLog() {
    actionLog.innerHTML = '<div class="log-placeholder">Click "Step" or "Auto-play" to start</div>';
    roundNumber = 0;
}

