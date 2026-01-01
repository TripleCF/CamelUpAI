"""
Monte Carlo oracle for Camel Up opponent strategy.

This module provides a Monte Carlo simulation-based action selection
that evaluates all valid actions by simulating random game completions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from itertools import permutations, product

from .constants import (
    ACTION_ROLL,
    ACTION_PLACE_OASIS,
    ACTION_PLACE_MIRAGE,
    ACTION_GAME_WINNER_1ST,
    ACTION_GAME_LOSER_1ST,
    BET_ACTION_TO_RANK,
    GAME_WINNER_ACTION_TO_RANK,
    GAME_LOSER_ACTION_TO_RANK,
    CAMELS,
    NUM_ACTIONS,
    ROLL_REWARD,
    DICE_VALUES,
    GAME_END_BET_PAYOUTS,
    GAME_END_BET_PENALTY,
)
from .game_state import GameState


def simulate_random_completion(
    state: GameState,
    rng: np.random.Generator,
) -> None:
    """
    Simulate the rest of the leg with random actions for all players.
    
    Modifies state in-place. All players take uniformly random valid actions
    until the leg completes.
    
    Args:
        state: Game state to complete (modified in-place).
        rng: Random generator for reproducibility.
    """
    from .constants import NUM_PLAYERS
    
    current_player = 0
    
    while not state.is_leg_complete():
        valid_actions = state.get_valid_actions(current_player)
        
        if not valid_actions:
            # No valid actions (shouldn't happen in normal play)
            break
        
        action = rng.choice(valid_actions)
        execute_action(state, current_player, action, rng)
        
        current_player = (current_player + 1) % NUM_PLAYERS


def execute_action(
    state: GameState,
    player: int,
    action: int,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Execute an action for a player.
    
    Supports both leg actions (0-5) and full game actions (6-17).
    
    Args:
        state: Game state (modified in-place).
        player: Player index.
        action: Action index (0=roll, 1-5=leg bets, 6-7=desert, 8-17=game bets).
        rng: Random generator.
        
    Returns:
        Immediate reward for this action (ROLL_REWARD for rolling, 0 for others).
        
    Raises:
        ValueError: If action is invalid.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if action == ACTION_ROLL:
        if not state.dice_remaining:
            raise ValueError("Cannot roll: no dice remaining")
        state.roll_die_for_player(player, rng)
        return ROLL_REWARD
    
    elif action in BET_ACTION_TO_RANK:
        rank_idx = BET_ACTION_TO_RANK[action]
        rankings = state.get_rankings()
        camel = rankings[rank_idx]
        
        if not state.tiles_remaining[camel]:
            raise ValueError(f"Cannot bet on camel {camel}: no tiles remaining")
        state.take_bet_tile(player, camel)
        return 0
    
    elif action == ACTION_PLACE_OASIS:
        valid_spaces = state.can_place_desert_tile(player)
        if valid_spaces:
            leader_pos = state.get_leader_position()
            best = min(valid_spaces, key=lambda s: abs(s - leader_pos - 2))
            state.place_desert_tile(player, best, is_oasis=True)
        return 0
    
    elif action == ACTION_PLACE_MIRAGE:
        valid_spaces = state.can_place_desert_tile(player)
        if valid_spaces:
            leader_pos = state.get_leader_position()
            best = min(valid_spaces, key=lambda s: abs(s - leader_pos - 2))
            state.place_desert_tile(player, best, is_oasis=False)
        return 0
    
    elif action in GAME_WINNER_ACTION_TO_RANK:
        rank_idx = GAME_WINNER_ACTION_TO_RANK[action]
        rankings = state.get_rankings()
        camel = rankings[rank_idx]
        if camel in state.can_bet_game_winner(player):
            state.place_game_winner_bet(player, camel)
        return 0
    
    elif action in GAME_LOSER_ACTION_TO_RANK:
        rank_idx = GAME_LOSER_ACTION_TO_RANK[action]
        rankings = state.get_rankings()
        camel = rankings[rank_idx]
        if camel in state.can_bet_game_loser(player):
            state.place_game_loser_bet(player, camel)
        return 0
    
    else:
        raise ValueError(f"Invalid action: {action}")


def get_leg_probabilities(
    state: GameState, 
    simulations: int = 1000,
    rng: Optional[np.random.Generator] = None
) -> dict[int, tuple[float, float]]:
    """
    Calculate probability of each camel finishing 1st and 2nd in the current leg.
    
    Uses exact calculation if 3 or fewer dice remain, otherwise uses Monte Carlo.
    """
    dice_count = len(state.dice_remaining)
    
    if dice_count == 0:
        rankings = state.get_rankings()
        return {c: (1.0, 0.0) if c == rankings[0] else (0.0, 1.0) if c == rankings[1] else (0.0, 0.0) for c in CAMELS}
    
    if dice_count <= 3:
        # Exact calculation
        dice_remaining = list(state.dice_remaining)
        n = len(dice_remaining)
        rolls = list(product(DICE_VALUES, repeat=n))
        orderings = list(permutations(dice_remaining))
        
        win_counts = {c: 0 for c in CAMELS}
        second_counts = {c: 0 for c in CAMELS}
        total = len(rolls) * len(orderings)
        
        for ordering in orderings:
            for roll_values in rolls:
                sim_state = state.copy()
                for camel, distance in zip(ordering, roll_values):
                    sim_state.move_camel(camel, distance)
                rankings = sim_state.get_rankings()
                win_counts[rankings[0]] += 1
                second_counts[rankings[1]] += 1
        
        return {c: (win_counts[c] / total, second_counts[c] / total) for c in CAMELS}
    
    else:
        # Monte Carlo
        if rng is None:
            rng = np.random.default_rng()
            
        win_counts = {c: 0 for c in CAMELS}
        second_counts = {c: 0 for c in CAMELS}
        
        for _ in range(simulations):
            sim_state = state.copy()
            while not sim_state.is_leg_complete():
                sim_state.roll_die(rng)
            rankings = sim_state.get_rankings()
            win_counts[rankings[0]] += 1
            second_counts[rankings[1]] += 1
            
        return {c: (win_counts[c] / simulations, second_counts[c] / simulations) for c in CAMELS}


def get_game_probabilities(
    state: GameState,
    simulations: int = 1000,
    rng: Optional[np.random.Generator] = None
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Calculate probability of each camel winning or losing the entire game.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    win_counts = {c: 0 for c in CAMELS}
    lose_counts = {c: 0 for c in CAMELS}
    
    for _ in range(simulations):
        sim_state = state.copy()
        # Simple random completion until game end
        while not sim_state.is_game_complete():
            if sim_state.is_leg_complete():
                sim_state.start_new_leg()
            sim_state.roll_die(rng)
        
        rankings = sim_state.get_rankings()
        win_counts[rankings[0]] += 1
        lose_counts[rankings[-1]] += 1
        
    return (
        {c: win_counts[c] / simulations for c in CAMELS},
        {c: lose_counts[c] / simulations for c in CAMELS}
    )


def monte_carlo_action(
    state: GameState,
    player: int,
    simulations: int = 50,
    rng: Optional[np.random.Generator] = None,
    return_values: bool = False,
) -> int | tuple[int, dict[int, float]]:
    """
    Select the best action using an optimized probabilistic approach.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    valid_actions = state.get_valid_actions(player)
    if not valid_actions:
        return (0, {}) if return_values else 0
    if len(valid_actions) == 1:
        return (valid_actions[0], {valid_actions[0]: 0.0}) if return_values else valid_actions[0]

    # Pre-calculate probabilities
    leg_probs = get_leg_probabilities(state, simulations, rng)
    
    # Game-end bet optimization: only simulate if we are at least mid-way or occasionally for "foresight"
    leader_pos = state.get_leader_position()
    game_probs = None
    if leader_pos >= 11 or rng.random() < 0.2:
        game_probs = get_game_probabilities(state, simulations, rng)
    
    # Calculate Riskiness Factor
    # (Simplified version: lower coins relative to leader = higher riskiness)
    max_coins = max(state.player_coins)
    my_coins = state.player_coins[player]
    coin_delta = max_coins - my_coins
    # Riskiness goes from 0 to 1 based on gap (e.g., 12 coin gap = max risk)
    riskiness = min(1.0, max(0.0, coin_delta / 12.0))
    
    action_values: dict[int, float] = {}
    
    for action in valid_actions:
        ev = 0.0
        
        if action == ACTION_ROLL:
            # EV for rolling: 1 coin reward 
            ev = 1.0
            
        elif action in BET_ACTION_TO_RANK:
            # EV for leg bet
            rank_idx = BET_ACTION_TO_RANK[action]
            current_rankings = state.get_rankings()
            camel = current_rankings[rank_idx]
            
            p1, p2 = leg_probs[camel]
            p_other = 1.0 - p1 - p2
            
            tile_val = state.get_top_tile(camel)
            if tile_val == 0:
                ev = -10.0 # Should be filtered by get_valid_actions
            else:
                base_ev = (p1 * tile_val) + (p2 * 1.0) + (p_other * -1.0)
                # Apply riskiness: if risky, favor high payout (upshot)
                upshot = tile_val
                ev = base_ev + riskiness * (upshot - base_ev)
                
        elif action == ACTION_PLACE_OASIS or action == ACTION_PLACE_MIRAGE:
            # Heuristic: Desert tiles earn 1 coin per landing.
            # They are more valuable early in the leg when more camels are yet to move.
            dice_remaining = len(state.dice_remaining)
            ev = 0.4 + (dice_remaining * 0.12) # Ranges from ~0.5 to ~1.0
            
        elif action in GAME_WINNER_ACTION_TO_RANK:
            if game_probs:
                camel_id = state.get_rankings()[GAME_WINNER_ACTION_TO_RANK[action]]
                p_win = game_probs[0][camel_id]
                # Payout estimation: depends on order in stack for THAT camel
                existing_bets_on_camel = sum(1 for p, c in state.game_winner_bets if c == camel_id)
                payout = GAME_END_BET_PAYOUTS[min(existing_bets_on_camel, len(GAME_END_BET_PAYOUTS)-1)]
                
                base_ev = p_win * payout + (1.0 - p_win) * GAME_END_BET_PENALTY
                upshot = 8.0 # Best possible payout
                ev = base_ev + riskiness * (upshot - base_ev)
            else:
                # If we skipped simulation, give it a very low EV so it doesn't get picked randomly,
                # but it stays above 'invalid' actions.
                ev = -1.5 
                
        elif action in GAME_LOSER_ACTION_TO_RANK:
            if game_probs:
                camel_id = state.get_rankings()[GAME_LOSER_ACTION_TO_RANK[action]]
                p_lose = game_probs[1][camel_id]
                existing_bets_on_camel = sum(1 for p, c in state.game_loser_bets if c == camel_id)
                payout = GAME_END_BET_PAYOUTS[min(existing_bets_on_camel, len(GAME_END_BET_PAYOUTS)-1)]
                
                base_ev = p_lose * payout + (1.0 - p_lose) * GAME_END_BET_PENALTY
                upshot = 8.0
                ev = base_ev + riskiness * (upshot - base_ev)
            else:
                ev = -1.5
        
        action_values[action] = ev

    best_action = max(action_values, key=lambda a: action_values[a])
    
    if return_values:
        return best_action, action_values
    return best_action


class MCOpponent:
    """
    Monte Carlo-based opponent for the Camel Up environment.
    
    Wraps the monte_carlo_action function to provide a consistent
    interface for opponent action selection.
    
    Attributes:
        simulations: Number of MC simulations per action evaluation.
        rng: Random generator for reproducibility.
    """
    
    def __init__(
        self, 
        simulations: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Initialize MC opponent.
        
        Args:
            simulations: Number of simulations per probability batch.
            seed: Random seed for reproducibility.
        """
        self.simulations = simulations
        self.rng = np.random.default_rng(seed)
    
    def get_action(self, state: GameState, player: int) -> int:
        """
        Get the best action for a player using MC simulation.
        """
        return monte_carlo_action(
            state, 
            player, 
            simulations=self.simulations,
            rng=self.rng,
        )
    
    def get_action_with_values(
        self, 
        state: GameState, 
        player: int,
    ) -> tuple[int, dict[int, float]]:
        """
        Get best action and all action values.
        """
        return monte_carlo_action(
            state,
            player,
            simulations=self.simulations,
            rng=self.rng,
            return_values=True,
        )

    @staticmethod
    def exact_win_probabilities(state: GameState) -> dict[int, tuple[float, float]]:
        """Berechnet exakte P(1st) und P(2nd) für jedes Kamel."""
        return get_leg_probabilities(state)


class RandomOpponent:
    """
    Simple random opponent for baseline comparison.
    
    Selects uniformly at random from valid actions.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def get_action(self, state: GameState, player: int) -> int:
        """Get a random valid action."""
        valid_actions = state.get_valid_actions(player)
        if not valid_actions:
            return 0
        return self.rng.choice(valid_actions)
