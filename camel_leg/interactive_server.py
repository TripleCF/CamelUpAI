"""
Interactive scenario builder for Camel Up.

A Flask-based web interface for experimenting with custom game states
and seeing trained model predictions in real-time.
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from .constants import (
    ACTION_ROLL,
    ACTION_PLACE_OASIS,
    ACTION_PLACE_MIRAGE,
    ACTION_GAME_WINNER_1ST,
    ACTION_GAME_WINNER_2ND,
    ACTION_GAME_WINNER_3RD,
    ACTION_GAME_WINNER_4TH,
    ACTION_GAME_WINNER_5TH,
    ACTION_GAME_LOSER_1ST,
    ACTION_GAME_LOSER_2ND,
    ACTION_GAME_LOSER_3RD,
    ACTION_GAME_LOSER_4TH,
    ACTION_GAME_LOSER_5TH,
    ACTION_GAME_LOSER_5TH,
    BET_ACTION_TO_RANK,
    GAME_WINNER_ACTION_TO_RANK,
    GAME_LOSER_ACTION_TO_RANK,
    CAMEL_NAMES,
    CAMELS,
    NUM_ACTIONS,
    NUM_FULL_GAME_ACTIONS,
    NUM_CAMELS,
    NUM_PLAYERS,
    NUM_SPACES,
    OASIS_EFFECT,
    MIRAGE_EFFECT,
    GAME_END_BET_PAYOUTS,
    GAME_END_BET_PENALTY,
)
from .game_state import GameState
from .monte_carlo import MCOpponent, execute_action


app = Flask(__name__, static_folder='static', static_url_path='')


# Global state
_model = None
_current_state: Optional[GameState] = None
_rng = np.random.default_rng(None)


def load_model(model_path: str):
    """Load a trained MaskablePPO model."""
    global _model
    try:
        from sb3_contrib import MaskablePPO
        _model = MaskablePPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        _model = None


def state_to_dict(state: GameState, full_game: bool = True) -> dict[str, Any]:
    """Convert GameState to JSON-serializable dict."""
    # Build board representation (convert numpy int64 to native int)
    board = []
    for space_idx in range(NUM_SPACES):
        stack = state.board[space_idx]
        board.append({
            "space": int(space_idx),
            "camels": [{"id": int(c), "name": CAMEL_NAMES[int(c)]} for c in stack]
        })
    
    # Get rankings
    rankings = state.get_rankings()
    
    result = {
        "board": board,
        "diceRemaining": [{"id": int(c), "name": CAMEL_NAMES[int(c)]} for c in sorted(state.dice_remaining)],
        "tilesRemaining": {
            CAMEL_NAMES[int(c)]: [int(t) for t in tiles] for c, tiles in state.tiles_remaining.items()
        },
        "playerBets": [
            [{"camel": CAMEL_NAMES[int(c)], "value": int(v)} for c, v in bets]
            for bets in state.player_bets
        ],
        "playerCoins": [int(c) for c in state.player_coins],
        "rankings": [{"rank": i+1, "camel": CAMEL_NAMES[int(c)]} for i, c in enumerate(rankings)],
        "isLegComplete": bool(state.is_leg_complete()),
        "validActions": [int(a) for a in state.get_valid_actions(0)],  # RL agent actions
    }
    
    # Add full game state if requested
    if full_game:
        # Desert tiles: {space: {owner, type}}
        desert_tiles = {}
        if hasattr(state, 'desert_tiles') and state.desert_tiles:
            for space, (owner, is_oasis) in state.desert_tiles.items():
                desert_tiles[int(space)] = {
                    "owner": int(owner),
                    "type": "oasis" if is_oasis else "mirage"
                }
        
        # Game-end bets per player
        game_winner_bets_list = [[] for _ in range(NUM_PLAYERS)]
        game_loser_bets_list = [[] for _ in range(NUM_PLAYERS)]
        
        if hasattr(state, 'game_winner_bets'):
            # Check if it's the new format (list of tuples)
            if state.game_winner_bets and isinstance(state.game_winner_bets[0], tuple):
                for p, c in state.game_winner_bets:
                    if 0 <= p < NUM_PLAYERS:
                        game_winner_bets_list[p].append(int(c))
            # Fallback for old format (list of lists) just in case
            elif isinstance(state.game_winner_bets, list) and state.game_winner_bets and isinstance(state.game_winner_bets[0], list):
                for p, bets in enumerate(state.game_winner_bets):
                    if 0 <= p < NUM_PLAYERS:
                        game_winner_bets_list[p] = [int(c) for c in bets]
        
        if hasattr(state, 'game_loser_bets'):
            if state.game_loser_bets and isinstance(state.game_loser_bets[0], tuple):
                for p, c in state.game_loser_bets:
                    if 0 <= p < NUM_PLAYERS:
                        game_loser_bets_list[p].append(int(c))
            elif isinstance(state.game_loser_bets, list) and state.game_loser_bets and isinstance(state.game_loser_bets[0], list):
                for p, bets in enumerate(state.game_loser_bets):
                    if 0 <= p < NUM_PLAYERS:
                        game_loser_bets_list[p] = [int(c) for c in bets]
        
        result.update({
            "currentLeg": int(getattr(state, 'current_leg', 1)),
            "gameComplete": bool(state.is_game_complete()) if hasattr(state, 'is_game_complete') else False,
            "leaderPosition": int(state.get_leader_position()) if hasattr(state, 'get_leader_position') else 0,
            "desertTiles": desert_tiles,
            "gameWinnerBets": game_winner_bets_list,
            "gameLoserBets": game_loser_bets_list,
            "numPlayers": int(getattr(state, 'num_players', NUM_PLAYERS)),
        })
    
    return result


def dict_to_state(data: dict) -> GameState:
    """Create GameState from JSON dict."""
    state = GameState()
    
    # Parse board
    state.board = [[] for _ in range(NUM_SPACES)]
    for space_data in data.get("board", []):
        space_idx = space_data["space"]
        for camel_data in space_data["camels"]:
            camel_id = camel_data["id"] if isinstance(camel_data, dict) else camel_data
            state.board[space_idx].append(camel_id)
    
    # Parse dice remaining
    dice_data = data.get("diceRemaining", [])
    if dice_data:
        state.dice_remaining = set(
            d["id"] if isinstance(d, dict) else d for d in dice_data
        )
    else:
        state.dice_remaining = set(CAMELS)
    
    # Parse tiles remaining
    tiles_data = data.get("tilesRemaining", {})
    if tiles_data:
        state.tiles_remaining = {}
        for camel in CAMELS:
            name = CAMEL_NAMES[camel]
            if name in tiles_data:
                state.tiles_remaining[camel] = list(tiles_data[name])
            else:
                state.tiles_remaining[camel] = [5, 3, 2, 1]
    
    # Parse player bets and coins
    state.player_bets = data.get("playerBets", [[] for _ in range(NUM_PLAYERS)])
    state.player_coins = data.get("playerCoins", [0] * NUM_PLAYERS)
    
    # Parse full game state if present
    state.current_leg = data.get("currentLeg", 1)
    state.num_players = data.get("numPlayers", NUM_PLAYERS)
    
    # Parse desert tiles: {"space": {"owner": int, "type": str}}
    desert_data = data.get("desertTiles", {})
    state.desert_tiles = {}
    for space_str, tile_info in desert_data.items():
        space = int(space_str)
        owner = tile_info["owner"]
        is_oasis = tile_info["type"] == "oasis"
        state.desert_tiles[space] = (owner, is_oasis)
    
    # Parse game-end bets
    # Parse game-end bets
    # JSON provides list of lists (per player). Convert to global FIFO list.
    # We approximate order by iterating players 0->N, similar to how it was before failure.
    state.game_winner_bets = []
    raw_winner_bets = data.get("gameWinnerBets", [])
    if raw_winner_bets:
        for p, bets in enumerate(raw_winner_bets):
            for c in bets:
                state.game_winner_bets.append((p, int(c)))

    state.game_loser_bets = []
    raw_loser_bets = data.get("gameLoserBets", [])
    if raw_loser_bets:
        for p, bets in enumerate(raw_loser_bets):
            for c in bets:
                state.game_loser_bets.append((p, int(c)))
    
    return state


def get_model_prediction(state: GameState) -> dict[str, Any]:
    """Get the model's action and probabilities for a state (supports full game models)."""
    if _model is None:
        return {"error": "No model loaded"}
    
    try:
        from .full_game_env import CamelFullGameEnv
        
        # Create a temporary env to get observation
        env = CamelFullGameEnv(num_opponents=3)
        env.game_state = state
        env.agent_index = 0
        
        obs = env._get_observation()
        action_masks = env.action_masks()
        
        # Get action from model
        action, _ = _model.predict(obs, deterministic=True, action_masks=action_masks)
        action = int(action)
        
        # Get action probabilities (if available)
        action_probs = [0.0] * NUM_FULL_GAME_ACTIONS
        try:
            import torch
            obs_tensor = {}
            for key, value in obs.items():
                obs_tensor[key] = torch.tensor(value).unsqueeze(0).float()
            
            with torch.no_grad():
                dist = _model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.squeeze().numpy()
                action_probs = probs.tolist()
        except Exception:
            # Fall back to just showing chosen action
            action_probs[action] = 1.0
        
        # Get rankings for resolving action names
        rankings = state.get_rankings()
        
        def get_action_name(act: int) -> str:
            """Get human-readable action name."""
            if act == ACTION_ROLL:
                return "Roll"
            elif act in BET_ACTION_TO_RANK:
                rank_idx = BET_ACTION_TO_RANK[act]
                camel_id = rankings[rank_idx]
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                return f"Leg Bet {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
            elif act == ACTION_PLACE_OASIS:
                return "Place Oasis (+1)"
            elif act == ACTION_PLACE_MIRAGE:
                return "Place Mirage (-1)"
            elif act in GAME_WINNER_ACTION_TO_RANK:
                rank_idx = GAME_WINNER_ACTION_TO_RANK[act]
                camel_id = rankings[rank_idx]
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                return f"Winner {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
            elif act in GAME_LOSER_ACTION_TO_RANK:
                rank_idx = GAME_LOSER_ACTION_TO_RANK[act]
                camel_id = rankings[rank_idx]
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                return f"Loser {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
            return f"Action {act}"
        
        # Format action name
        action_name = get_action_name(action)
        
        # Format probabilities for all 18 actions
        formatted_probs = []
        for i, p in enumerate(action_probs):
            formatted_probs.append({
                "action": i, 
                "name": get_action_name(i), 
                "prob": p
            })

        return {
            "action": action,
            "actionName": action_name,
            "actionProbabilities": formatted_probs,
            "validActions": [bool(m) for m in action_masks],
        }
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}


def get_mc_recommendation(state: GameState, simulations: int = 100) -> dict[str, Any]:
    """Get Monte Carlo's recommended action using the optimized engine."""
    try:
        opponent = MCOpponent(simulations=simulations)
        valid_actions = state.get_valid_actions(0)
        
        if not valid_actions:
            return {"error": "No valid actions"}
            
        best_action, values = opponent.get_action_with_values(state, 0)
        
        rankings = state.get_rankings()
        rank_based_evs = []
        mapped_best_action = best_action

        for action in range(NUM_ACTIONS):
            name = ""
            if action == ACTION_ROLL:
                name = "Roll"
            elif action in BET_ACTION_TO_RANK:
                rank_idx = BET_ACTION_TO_RANK[action]
                camel_id = rankings[rank_idx]
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                name = f"Bet {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
            
            rank_based_evs.append({
                "action": action,
                "name": name,
                "immediateEv": round(values.get(action, 0.0), 2),
                "strategicEv": round(values.get(action, 0.0), 2),
                "isValid": action in valid_actions,
            })

        # Format action name for best action
        action_name = rank_based_evs[mapped_best_action]["name"] if mapped_best_action < len(rank_based_evs) else f"Action {mapped_best_action}"
        
        return {
            "action": mapped_best_action,
            "actionName": action_name,
            "expectedValues": rank_based_evs,
        }
    except Exception as e:
        return {"error": str(e)}


def get_mc_recommendation_full_game(state: GameState, simulations: int = 100) -> dict[str, Any]:
    """Get Monte Carlo EVs for full game actions using the optimized engine."""
    try:
        # Use MCOpponent to get values for all actions
        # We use a higher simulation count if requested to ensure stability in UI
        opponent = MCOpponent(simulations=max(simulations, 500))
        valid_actions = state.get_valid_actions(0)
        
        # We temporarily force game probabilities to be calculated for the UI recommendation
        # by passing a state where the leader is "ahead" if needed, but better to 
        # just have a way to force it. Let's use the underlying functions directly.
        from .monte_carlo import get_leg_probabilities, get_game_probabilities
        leg_probs = get_leg_probabilities(state, simulations)
        game_probs = get_game_probabilities(state, simulations) # Force for UI
        
        # Now we can manually construct the EVs similar to how monte_carlo_action does
        # but with full information.
        rankings = state.get_rankings()
        action_evs = []
        
        # Riskiness based on player standing
        max_coins = max(state.player_coins)
        my_coins = state.player_coins[0]
        riskiness = min(1.0, max(0.0, (max_coins - my_coins) / 12.0))
        
        for action in range(NUM_FULL_GAME_ACTIONS):
            ev = -10.0
            name = f"Action {action}"
            category = "other"
            is_valid = action in valid_actions
            
            if action == ACTION_ROLL:
                ev = 1.0
                name = "Roll"
                category = "dice"
            elif action in BET_ACTION_TO_RANK:
                rank_idx = BET_ACTION_TO_RANK[action]
                camel_id = rankings[rank_idx]
                p1, p2 = leg_probs[camel_id]
                p_other = 1.0 - p1 - p2
                tile_val = state.get_top_tile(camel_id)
                base_ev = (p1 * tile_val) + (p2 * 1.0) + (p_other * -1.0) if tile_val > 0 else -1.0
                ev = base_ev + riskiness * (tile_val - base_ev) if tile_val > 0 else base_ev
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                name = f"Leg Bet {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
                category = "leg_bet"
            elif action == ACTION_PLACE_OASIS or action == ACTION_PLACE_MIRAGE:
                dice_remaining = len(state.dice_remaining)
                ev = 0.4 + (dice_remaining * 0.12)
                name = "Place Oasis (+1)" if action == ACTION_PLACE_OASIS else "Place Mirage (-1)"
                category = "desert"
            elif action in GAME_WINNER_ACTION_TO_RANK:
                rank_idx = GAME_WINNER_ACTION_TO_RANK[action]
                camel_id = rankings[rank_idx]
                p_win = game_probs[0][camel_id]
                existing_bets = sum(1 for p, c in state.game_winner_bets if c == camel_id)
                payout = GAME_END_BET_PAYOUTS[min(existing_bets, len(GAME_END_BET_PAYOUTS)-1)]
                base_ev = p_win * payout + (1.0 - p_win) * GAME_END_BET_PENALTY
                ev = base_ev + riskiness * (8.0 - base_ev)
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                name = f"Winner {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
                category = "game_winner"
            elif action in GAME_LOSER_ACTION_TO_RANK:
                rank_idx = GAME_LOSER_ACTION_TO_RANK[action]
                camel_id = rankings[rank_idx]
                p_lose = game_probs[1][camel_id]
                existing_bets = sum(1 for p, c in state.game_loser_bets if c == camel_id)
                payout = GAME_END_BET_PAYOUTS[min(existing_bets, len(GAME_END_BET_PAYOUTS)-1)]
                base_ev = p_lose * payout + (1.0 - p_lose) * GAME_END_BET_PENALTY
                ev = base_ev + riskiness * (8.0 - base_ev)
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                name = f"Loser {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
                category = "game_loser"
            
            action_evs.append({
                "action": action,
                "name": name,
                "category": category,
                "ev": round(ev, 2),
                "isValid": is_valid,
            })
            
        best = max([a for a in action_evs if a["isValid"]], key=lambda x: x["ev"])
        
        return {
            "action": best["action"],
            "actionName": best["name"],
            "expectedValues": action_evs,
            "gameWinProbs": {CAMEL_NAMES[c]: round(p, 3) for c, p in game_probs[0].items()},
            "gameLoseProbs": {CAMEL_NAMES[c]: round(p, 3) for c, p in game_probs[1].items()},
        }
    except Exception as e:
        import traceback
        return {"error": f"{str(e)}\n{traceback.format_exc()}"}


def run_win_simulations(state: GameState, num_sims: int = 500) -> dict[str, Any]:
    """Run simulations to get win probability distribution for each camel."""
    try:
        win_counts = {c: 0 for c in CAMELS}
        second_counts = {c: 0 for c in CAMELS}
        
        for _ in range(num_sims):
            sim_state = state.copy()
            rng = np.random.default_rng()
            
            # Simulate until GAME complete
            while not sim_state.is_game_complete():
                if sim_state.is_leg_complete():
                    sim_state.start_new_leg()
                
                # Check because start_new_leg might not be enough if no dice? 
                # Actually we just roll until game ends.
                if not sim_state.is_leg_complete():
                    sim_state.roll_die(rng)
                else: 
                     # Should not happen in loop unless game end check is after
                     break

            rankings = sim_state.get_rankings()
            win_counts[rankings[0]] += 1
            second_counts[rankings[1]] += 1
            # Note: second_counts here means 2nd in Game, not Leg.
        
        results = []
        for camel in CAMELS:
            results.append({
                "camelId": camel,
                "camelName": CAMEL_NAMES[camel],
                "winProb": win_counts[camel] / num_sims,
                "secondProb": second_counts[camel] / num_sims,
                "winCount": win_counts[camel],
                "secondCount": second_counts[camel],
            })
        
        # Sort by win probability
        results.sort(key=lambda x: x["winProb"], reverse=True)
        
        return {
            "simulations": num_sims,
            "results": results,
        }
    except Exception as e:
        return {"error": str(e)}


# Routes

@app.route('/')
def index():
    """Serve main page."""
    return send_from_directory('static', 'index.html')


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current game state."""
    global _current_state
    if _current_state is None:
        _current_state = GameState.create_random_start(_rng)
    return jsonify(state_to_dict(_current_state))


@app.route('/api/state', methods=['POST'])
def set_state():
    """Set game state from JSON."""
    global _current_state
    data = request.get_json()
    _current_state = dict_to_state(data)
    return jsonify(state_to_dict(_current_state))


@app.route('/api/reset', methods=['POST'])
def reset_state():
    """Reset to a new random state."""
    global _current_state
    _current_state = GameState.create_random_start(_rng)
    return jsonify(state_to_dict(_current_state))


@app.route('/api/predict', methods=['POST'])
def predict():
    """Get model prediction for current/provided state."""
    global _current_state
    
    data = request.get_json() or {}
    if "board" in data:
        state = dict_to_state(data)
    else:
        if _current_state is None:
            _current_state = GameState.create_random_start(_rng)
        state = _current_state
    
    return jsonify(get_model_prediction(state))


@app.route('/api/monte-carlo', methods=['POST'])
def monte_carlo():
    """Get Monte Carlo recommendation."""
    global _current_state
    
    data = request.get_json() or {}
    if "board" in data:
        state = dict_to_state(data)
    else:
        if _current_state is None:
            _current_state = GameState.create_random_start(_rng)
        state = _current_state
    
    simulations = data.get("simulations", 100)
    return jsonify(get_mc_recommendation(state, simulations))


@app.route('/api/monte-carlo-full', methods=['POST'])
def monte_carlo_full():
    """Get Monte Carlo recommendation for full game (18 actions)."""
    global _current_state
    
    data = request.get_json() or {}
    if "board" in data:
        state = dict_to_state(data)
    else:
        if _current_state is None:
            _current_state = GameState.create_random_start(_rng)
        state = _current_state
    
    simulations = data.get("simulations", 100)
    return jsonify(get_mc_recommendation_full_game(state, simulations))


@app.route('/api/simulate-wins', methods=['POST'])
def simulate_wins():
    """Run win probability simulations."""
    global _current_state
    
    data = request.get_json() or {}
    if "board" in data:
        state = dict_to_state(data)
    else:
        if _current_state is None:
            _current_state = GameState.create_random_start(_rng)
        state = _current_state
    
    num_sims = data.get("simulations", 500)
    return jsonify(run_win_simulations(state, num_sims))


@app.route('/api/roll', methods=['POST'])
def roll_dice():
    """Roll a die in the current state."""
    global _current_state
    
    if _current_state is None:
        _current_state = GameState.create_random_start(_rng)
    
    if _current_state.is_leg_complete():
        return jsonify({"error": "Leg is complete, no dice remaining"})
    
    camel, distance = _current_state.roll_die(_rng)
    
    return jsonify({
        "camel": CAMEL_NAMES[int(camel)],
        "camelId": int(camel),
        "distance": int(distance),
        "state": state_to_dict(_current_state),
    })


@app.route('/api/bet', methods=['POST'])
def place_bet():
    """Place a bet on a camel."""
    global _current_state
    
    if _current_state is None:
        _current_state = GameState.create_random_start(_rng)
    
    data = request.get_json() or {}
    camel_id = data.get("camelId", 0)
    player = data.get("player", 0)
    
    try:
        tile_value = _current_state.take_bet_tile(player, camel_id)
        return jsonify({
            "camel": CAMEL_NAMES[camel_id],
            "tileValue": tile_value,
            "state": state_to_dict(_current_state),
        })
    except ValueError as e:
        return jsonify({"error": str(e)})


@app.route('/api/simulate-step', methods=['POST'])
def simulate_step():
    """Run one simulation step: single player action."""
    global _current_state, _current_player
    
    # Initialize player tracking if needed
    if not hasattr(simulate_step, 'current_player'):
        simulate_step.current_player = 0
    
    if _current_state is None:
        _current_state = GameState.create_random_start(_rng)
        simulate_step.current_player = 0
    
    if _current_state.is_leg_complete():
        # Score bets if leg just completed
        bet_deltas = _current_state.score_bets()
        simulate_step.current_player = 0  # Reset for next leg
        return jsonify({
            "legComplete": True,
            "betResults": {
                f"Player {p}": delta for p, delta in bet_deltas.items()
            },
            "finalRankings": [CAMEL_NAMES[c] for c in _current_state.get_rankings()],
            "state": state_to_dict(_current_state),
            "actions": [],
            "currentPlayer": simulate_step.current_player,
        })
    
    from .monte_carlo import MCOpponent
    from .full_game_env import CamelFullGameEnv
    
    player = simulate_step.current_player
    action = 0
    action_name = "Unknown"
    
    # Determine action based on player
    if player == 0:
        # Player 0 (RL Agent) - use model if available, else MC
        if _model is not None:
            # Try to use FullGameEnv by default as it supports all actions
            try:
                env = CamelFullGameEnv(num_opponents=3)
                env.game_state = _current_state
                env.agent_index = 0
                
                obs = env._get_observation()
                action_masks = env.action_masks()
                action, _ = _model.predict(obs, deterministic=True, action_masks=action_masks)
                action = int(action)
            except Exception as e:
                print(f"Error getting model prediction (full game): {e}")
                # Fallback to MC
                mc = MCOpponent(simulations=50)
                action = mc.get_action(_current_state, 0)
            
            # Resolve name
            if action == ACTION_ROLL:
                action_name = "Roll"
            elif 1 <= action <= 5:
                rankings = _current_state.get_rankings()
                rank_idx = BET_ACTION_TO_RANK.get(action, 0)
                camel_id = rankings[rank_idx]
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                action_name = f"Bet {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
            elif action == ACTION_PLACE_OASIS:
                action_name = "Place Oasis (+1)"
            elif action == ACTION_PLACE_MIRAGE:
                action_name = "Place Mirage (-1)"
            elif 8 <= action <= 12:  # Game Winner
                rankings = _current_state.get_rankings()
                rank_idx = GAME_WINNER_ACTION_TO_RANK.get(action, 0)
                camel_id = rankings[rank_idx]
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                action_name = f"Winner Bet {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
            elif 13 <= action <= 17: # Game Loser
                rankings = _current_state.get_rankings()
                rank_idx = GAME_LOSER_ACTION_TO_RANK.get(action, 0)
                camel_id = rankings[rank_idx]
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                action_name = f"Loser Bet {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"

        else:
            mc = MCOpponent(simulations=100)
            action = mc.get_action(_current_state, 0)
            rankings = _current_state.get_rankings()
            
            if action == ACTION_ROLL:
                action_name = "Roll"
            elif 1 <= action <= 5:
                rank_idx = BET_ACTION_TO_RANK.get(action, 0)
                camel_id = rankings[rank_idx]
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                action_name = f"Leg Bet {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
            elif action == ACTION_PLACE_OASIS:
                action_name = "Place Oasis (+1)"
            elif action == ACTION_PLACE_MIRAGE:
                action_name = "Place Mirage (-1)"
            elif 8 <= action <= 12:
                rank_idx = GAME_WINNER_ACTION_TO_RANK.get(action, 0)
                camel_id = rankings[rank_idx]
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                action_name = f"Winner Bet {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
            elif 13 <= action <= 17:
                rank_idx = GAME_LOSER_ACTION_TO_RANK.get(action, 0)
                camel_id = rankings[rank_idx]
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
                action_name = f"Loser Bet {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
            else:
                action_name = f"Action {action}"
        player_name = "🤖 Agent"
    else:
        # Opponents use MC
        mc = MCOpponent(simulations=100)
        action = mc.get_action(_current_state, player)
        player_name = f"🎲 Bot {player}"
        rankings = _current_state.get_rankings()
        
        if action == ACTION_ROLL:
            action_name = "Roll"
        elif 1 <= action <= 5:
             rank_idx = BET_ACTION_TO_RANK.get(action, 0)
             camel_id = rankings[rank_idx]
             suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
             action_name = f"Leg Bet {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
        elif action == ACTION_PLACE_OASIS:
            action_name = "Place Oasis (+1)"
        elif action == ACTION_PLACE_MIRAGE:
            action_name = "Place Mirage (-1)"
        elif 8 <= action <= 12:
            rank_idx = GAME_WINNER_ACTION_TO_RANK.get(action, 0)
            camel_id = rankings[rank_idx]
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
            action_name = f"Winner Bet {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
        elif 13 <= action <= 17:
            rank_idx = GAME_LOSER_ACTION_TO_RANK.get(action, 0)
            camel_id = rankings[rank_idx]
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_idx + 1, "th")
            action_name = f"Loser Bet {rank_idx+1}{suffix} ({CAMEL_NAMES[camel_id]})"
        else:
             action_name = f"Action {action}"

    
    # Execute the action (Handling all 18 actions)
    try:
        rankings = _current_state.get_rankings()
        
        if action == ACTION_ROLL:
            if _current_state.dice_remaining:
                camel, distance = _current_state.roll_die_for_player(player, _rng)
                action_name = f"{action_name} → {CAMEL_NAMES[camel]} moved {distance}"
            else:
                action_name = "Roll (no dice left)"
                
        elif 1 <= action <= 5:
            # Leg Bet
            rank_idx = BET_ACTION_TO_RANK.get(action, 0)
            camel_id = rankings[rank_idx]
            
            tile_value = _current_state.get_top_tile(camel_id)
            if tile_value > 0:
                _current_state.take_bet_tile(player, camel_id)
                action_name = f"Bet {CAMEL_NAMES[camel_id]} (tile: {tile_value})"
            else:
                action_name = f"Bet {CAMEL_NAMES[camel_id]} (failed - no tiles)"
                
        elif action == ACTION_PLACE_OASIS:
            valid_spaces = _current_state.can_place_desert_tile(player)
            if valid_spaces:
                leader_pos = _current_state.get_leader_position()
                best = min(valid_spaces, key=lambda s: abs(s - leader_pos - 2))
                _current_state.place_desert_tile(player, best, is_oasis=True)
                action_name = f"Place Oasis at {best + 1}"
            else:
                action_name = "Place Oasis (failed - no valid space)"
                
        elif action == ACTION_PLACE_MIRAGE:
            valid_spaces = _current_state.can_place_desert_tile(player)
            if valid_spaces:
                leader_pos = _current_state.get_leader_position()
                best = min(valid_spaces, key=lambda s: abs(s - leader_pos - 2))
                _current_state.place_desert_tile(player, best, is_oasis=False)
                action_name = f"Place Mirage at {best + 1}"
            else:
                 action_name = "Place Mirage (failed - no valid space)"
                 
        elif 8 <= action <= 12:
            rank_idx = GAME_WINNER_ACTION_TO_RANK.get(action, 0)
            camel_id = rankings[rank_idx]
            if camel_id in _current_state.can_bet_game_winner(player):
                _current_state.place_game_winner_bet(player, camel_id)
                action_name = f"Winner Bet {CAMEL_NAMES[camel_id]}"
            else:
                action_name = f"Winner Bet {CAMEL_NAMES[camel_id]} (failed)"
                
        elif 13 <= action <= 17:
            rank_idx = GAME_LOSER_ACTION_TO_RANK.get(action, 0)
            camel_id = rankings[rank_idx]
            if camel_id in _current_state.can_bet_game_loser(player):
                _current_state.place_game_loser_bet(player, camel_id)
                action_name = f"Loser Bet {CAMEL_NAMES[camel_id]}"
            else:
                action_name = f"Loser Bet {CAMEL_NAMES[camel_id]} (failed)"
        
    except Exception as e:
        action_name += f" (ERROR: {str(e)})"
        print(f"Error executing action {action}: {e}")
    
    action_result = {
        "player": player,
        "playerName": player_name,
        "action": action,
        "actionName": action_name,
    }
    
    # Check if leg complete
    leg_complete = _current_state.is_leg_complete()
    
    # Check if game complete (camel crossed finish)
    game_complete = _current_state.is_game_complete() if hasattr(_current_state, 'is_game_complete') else False
    
    # Advance to next player
    simulate_step.current_player = (player + 1) % NUM_PLAYERS
    
    result = {
        "legComplete": leg_complete,
        "gameComplete": game_complete,
        "state": state_to_dict(_current_state),
        "actions": [action_result],
        "currentPlayer": simulate_step.current_player,
    }
    
    if leg_complete:
        bet_deltas = _current_state.score_bets()
        result["betResults"] = {f"Player {p}": delta for p, delta in bet_deltas.items()}
        result["finalRankings"] = [CAMEL_NAMES[c] for c in _current_state.get_rankings()]
        simulate_step.current_player = 0  # Reset for next leg
        
        if game_complete:
            # Game is over - score game-end bets
            if hasattr(_current_state, 'score_game_end_bets'):
                game_bet_deltas = _current_state.score_game_end_bets()
                result["gameBetResults"] = {f"Player {p}": delta for p, delta in game_bet_deltas.items()}
            result["finalCoins"] = [int(c) for c in _current_state.player_coins]
        else:
            # Start new leg
            if hasattr(_current_state, 'start_new_leg'):
                _current_state.start_new_leg()
                result["newLegStarted"] = True
                result["currentLeg"] = getattr(_current_state, 'current_leg', 1)
    
    return jsonify(result)


def run_server(model_path: Optional[str] = None, port: int = 5000, debug: bool = False):
    """Run the interactive server."""
    if model_path:
        load_model(model_path)
    
    print(f"\n🐪 Camel Up Interactive Scenario Builder")
    print(f"   Open http://localhost:{port} in your browser\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else None
    run_server(model)
