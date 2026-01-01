"""
Gymnasium environment for Camel Up single-leg simulation.

This module wraps the core game logic in a gymnasium-compatible interface
for training RL agents using algorithms like PPO.
"""

from __future__ import annotations

from typing import Any, Optional, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .constants import (
    ACTION_ROLL,
    BET_ACTION_TO_RANK,
    CAMELS,
    CAMEL_NAMES,
    INVALID_ACTION_PENALTY,
    NUM_ACTIONS,
    NUM_CAMELS,
    NUM_PLAYERS,
    NUM_SPACES,
    RL_AGENT_INDEX,
    ROLL_REWARD,
)
from .game_state import GameState
from .monte_carlo import MCOpponent, RandomOpponent, execute_action


class CamelLegEnv(gym.Env):
    """
    Gymnasium environment for a single leg of Camel Up.
    
    The RL agent is player 0, competing against 3 opponents.
    Each step() call executes actions for all 4 players (RL agent first,
    then opponents in order). The observation is the state after all
    4 players have acted.
    
    Opponents can use either Monte Carlo simulation (MCOpponent) or
    random action selection (RandomOpponent) for their strategies.
    
    Attributes:
        game_state: The current GameState object.
        opponent_type: "mc" for Monte Carlo opponents, "random" for random.
        opponent_simulations: Number of MC simulations per opponent action.
        render_mode: "human" for text output, None for no rendering.
        
    Example:
        >>> env = CamelLegEnv(opponent_type="mc", opponent_simulations=50)
        >>> obs, info = env.reset()
        >>> action = 0  # Roll
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        opponent_type: Literal["mc", "random"] = "mc",
        opponent_simulations: int = 5,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Camel Up environment.
        
        Args:
            opponent_type: "mc" for Monte Carlo opponents, "random" for random.
            opponent_simulations: Number of simulations for MC opponents.
            render_mode: "human" for text rendering, None for silent.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        
        self.opponent_type = opponent_type
        self.opponent_simulations = opponent_simulations
        self.render_mode = render_mode
        
        # Initialize random generator
        self._rng = np.random.default_rng(seed)
        
        # Initialize opponents
        if opponent_type == "mc":
            self._opponents = [
                MCOpponent(simulations=opponent_simulations, seed=self._rng.integers(0, 2**31))
                for _ in range(NUM_PLAYERS - 1)
            ]
        else:
            self._opponents = [
                RandomOpponent(seed=self._rng.integers(0, 2**31))
                for _ in range(NUM_PLAYERS - 1)
            ]
        
        # Define observation space (color-agnostic, rank-based)
        self.observation_space = spaces.Dict({
            # Positions of camels sorted by rank: [pos_1st, pos_2nd, ..., pos_5th]
            "ranked_positions": spaces.Box(
                low=0, high=15,
                shape=(NUM_CAMELS,),
                dtype=np.int8,
            ),
            # Dice rolled status by rank (1 if die rolled, 0 if still in pyramid)
            "dice_rolled_by_rank": spaces.MultiBinary(NUM_CAMELS),
            # Top available tile value by rank (0 if none)
            "tiles_available_by_rank": spaces.Box(
                low=0, high=5,
                shape=(NUM_CAMELS,),
                dtype=np.int8,
            ),
            # All players' coin totals
            "player_coins": spaces.Box(
                low=-100, high=100,
                shape=(NUM_PLAYERS,),
                dtype=np.int16,
            ),
        })
        
        # Action space: roll (0) or bet on rank (1-5)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
        # Game state (initialized in reset())
        self.game_state: Optional[GameState] = None
        
        # Current rankings for action translation
        self._current_rankings: list[int] = []
        
        # Track agent's coins at start of EPISODE for terminal reward calculation
        self._agent_coins_at_episode_start: int = 0
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment to a new random starting state.
        
        Args:
            seed: Random seed for this episode.
            options: Additional options (unused).
            
        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Create new game state with random camel placement
        self.game_state = GameState.create_random_start(rng=self._rng)
        
        self._agent_coins_at_episode_start = self.game_state.player_coins[RL_AGENT_INDEX]
        
        if self.render_mode == "human":
            self._render_state("=== NEW LEG STARTED ===")
        
        return self._get_observation(), self._get_info()
    
    def step(
        self, 
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Execute one step: RL agent action + 3 opponent actions.
        
        The RL agent always acts first, followed by opponents in order.
        The leg ends when all 5 dice have been rolled.
        
        Args:
            action: Action index (0=roll, 1-5=bet on rank).
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
            - reward: Coin delta for RL agent since last step.
            - terminated: True if leg is complete.
            - truncated: Always False (no time limit).
        """
        assert self.game_state is not None, "Must call reset() before step()"
        
        # Note: we don't update coins here - we track from episode start for terminal reward
        
        # Check if action is valid
        valid_actions = self.action_masks()
        if not valid_actions[action]:
            # Invalid action: return penalty immediately, no opponent actions
            if self.render_mode == "human":
                print(f"  [INVALID] Agent tried action {action}")
            return (
                self._get_observation(),
                float(INVALID_ACTION_PENALTY),
                False,  # not terminated
                False,  # not truncated
                self._get_info(),
            )
        
        # Translate rank-based action to color-based action for execution
        resolved_action = self._resolve_action(action)
        
        # Execute agent action
        execute_action(self.game_state, RL_AGENT_INDEX, resolved_action, self._rng)
        if self.render_mode == "human":
            self._render_action(RL_AGENT_INDEX, resolved_action)
        
        # Check if leg complete after agent action
        if self.game_state.is_leg_complete():
            return self._finalize_leg()
        
        # Execute opponent actions
        for opponent_idx, opponent in enumerate(self._opponents):
            player = opponent_idx + 1  # Players 1, 2, 3
            
            # Get opponent action
            opp_action = opponent.get_action(self.game_state, player)
            
            # Execute opponent action
            try:
                execute_action(self.game_state, player, opp_action, self._rng)
                if self.render_mode == "human":
                    self._render_action(player, opp_action)
            except ValueError:
                # Opponent made invalid action (shouldn't happen with get_valid_actions)
                if self.render_mode == "human":
                    print(f"  [ERROR] Opponent {player} invalid action {opp_action}")
            
            # Check if leg complete after each opponent
            if self.game_state.is_leg_complete():
                return self._finalize_leg()
        
        # Leg not complete, continue
        terminated = False
        truncated = False
        
        # Terminal-only rewards: return 0 for intermediate steps
        # This equalizes timing of roll (+1 immediate) vs bet (delayed) rewards
        return self._get_observation(), 0.0, terminated, truncated, self._get_info()
    
    def _finalize_leg(self) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Finalize the leg: score bets and return terminal state.
        
        Returns:
            Final (observation, reward, terminated=True, truncated=False, info).
        """
        # Score all bets
        bet_deltas = self.game_state.score_bets()
        
        if self.render_mode == "human":
            self._render_state("=== LEG COMPLETE ===")
            print(f"  Rankings: {[CAMEL_NAMES[c] for c in self.game_state.get_rankings()]}")
            print(f"  Bet payouts: {bet_deltas}")
            print(f"  Final coins: {self.game_state.player_coins}")
        
        # Terminal reward: total coin change for entire episode
        agent_coin_delta = (
            self.game_state.player_coins[RL_AGENT_INDEX]
            - self._agent_coins_at_episode_start
        )
        
        return (
            self._get_observation(),
            float(agent_coin_delta),
            True,  # terminated
            False,  # truncated
            self._get_info(),
        )
    
    def action_masks(self) -> np.ndarray:
        """
        Return boolean mask of valid actions.
        
        Required for MaskablePPO from sb3-contrib.
        Actions are rank-based: action 1 = bet on 1st place, etc.
        
        Returns:
            Boolean array of shape (6,) where True = valid action.
        """
        masks = np.zeros(NUM_ACTIONS, dtype=bool)
        
        if self.game_state is None:
            return masks
        
        # Roll is valid if dice remain
        if self.game_state.dice_remaining:
            masks[ACTION_ROLL] = True
        
        # Bet actions: check if tile available for camel at that rank
        rankings = self.game_state.get_rankings()
        for action, rank_idx in BET_ACTION_TO_RANK.items():
            camel = rankings[rank_idx]
            if self.game_state.tiles_remaining[camel]:
                masks[action] = True
        
        return masks
    
    def _resolve_action(self, action: int) -> int:
        """
        No resolution needed as execute_action now supports rank-based actions.
        """
        return action
    
    def _get_observation(self) -> dict[str, np.ndarray]:
        """
        Convert game state to rank-ordered observation dict.
        
        The observation is color-agnostic: camels are represented by their
        current ranking position, not by their color identity.
        
        Returns:
            Observation dictionary matching observation_space.
        """
        state = self.game_state
        
        # Get current rankings and cache for action translation
        rankings = state.get_rankings()
        self._current_rankings = rankings
        
        # Positions by rank [pos_1st, pos_2nd, ..., pos_5th]
        ranked_positions = np.array([
            state.get_camel_position(camel)[0]  # space index only
            for camel in rankings
        ], dtype=np.int8)
        
        # Dice rolled status by rank (1 if rolled/removed, 0 if still in pyramid)
        dice_rolled_by_rank = np.array([
            0 if camel in state.dice_remaining else 1
            for camel in rankings
        ], dtype=np.int8)
        
        # Tiles available by rank
        tiles_by_rank = np.array([
            state.get_top_tile(camel)
            for camel in rankings
        ], dtype=np.int8)
        
        # Player coins
        player_coins = np.array(state.player_coins, dtype=np.int16)
        
        return {
            "ranked_positions": ranked_positions,
            "dice_rolled_by_rank": dice_rolled_by_rank,
            "tiles_available_by_rank": tiles_by_rank,
            "player_coins": player_coins,
        }
    
    def _get_info(self) -> dict[str, Any]:
        """
        Return auxiliary info dict.
        
        Returns:
            Dict with rankings and valid_actions for debugging.
        """
        if self.game_state is None:
            return {}
        
        return {
            "rankings": self.game_state.get_rankings(),
            "valid_actions": self.game_state.get_valid_actions(RL_AGENT_INDEX),
            "dice_remaining": list(self.game_state.dice_remaining),
        }
    
    def render(self) -> None:
        """Render current state (if render_mode='human')."""
        if self.render_mode == "human" and self.game_state is not None:
            self._render_state("Current State")
    
    def _render_state(self, title: str) -> None:
        """Print game state."""
        print(f"\n{title}")
        print(self.game_state)
    
    def _render_action(self, player: int, action: int) -> None:
        """Print action taken."""
        if action == ACTION_ROLL:
            print(f"  P{player}: Rolled")
        elif action in BET_ACTION_TO_RANK:
            rank_idx = BET_ACTION_TO_RANK[action]
            camel = self.game_state.get_rankings()[rank_idx]
            print(f"  P{player}: Bet on {CAMEL_NAMES[camel]}")
    
    def close(self) -> None:
        """Clean up resources."""
        pass
