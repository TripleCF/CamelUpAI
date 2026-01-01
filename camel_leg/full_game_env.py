"""
Gymnasium environment for full Camel Up game.

Extends the single-leg environment to support:
- Multiple legs until a camel crosses the finish line
- Desert tiles (oasis/mirage)
- Game-end winner/loser bets
- Variable number of opponents
- Win-based rewards
"""

from __future__ import annotations

from typing import Any, Optional, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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
    CAMEL_NAMES,
    INVALID_ACTION_PENALTY,
    NUM_CAMELS,
    NUM_FULL_GAME_ACTIONS,
    NUM_SPACES,
    OASIS_EFFECT,
    MIRAGE_EFFECT,
)
from .game_state import GameState
from .monte_carlo import MCOpponent, RandomOpponent, execute_action


class CamelFullGameEnv(gym.Env):
    """
    Gymnasium environment for a full Camel Up game.
    
    The RL agent plays against configurable opponents in a multi-leg game
    until a camel crosses the finish line.
    
    Key differences from CamelLegEnv:
    - Multiple legs until game ends
    - Desert tile placement actions
    - Game-end winner/loser bets
    - Agent doesn't always go first (random starting position)
    - Win-based rewards (did agent finish with most coins?)
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        num_opponents: int = 3,
        opponent_type: Literal["mc", "random", "mixed"] = "mc",
        opponent_simulations: int = 50,
        reward_mode: Literal["win", "coins", "combined"] = "win",
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the full game environment.
        
        Args:
            num_opponents: Number of opponents (1-7).
            opponent_type: Strategy for opponents.
            opponent_simulations: MC simulations per opponent action.
            reward_mode: "win" for +1/-1, "coins" for coin delta, "combined".
            render_mode: "human" for text output, None for silent.
            seed: Random seed.
        """
        super().__init__()
        
        self.num_opponents = num_opponents
        self.num_players = num_opponents + 1
        self.opponent_type = opponent_type
        self.opponent_simulations = opponent_simulations
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        
        # Initialize random generator
        self._rng = np.random.default_rng(seed)
        
        # Agent index (randomized each game)
        self.agent_index = 0
        
        # Initialize opponents
        self._create_opponents()
        
        # Define observation space
        self.observation_space = spaces.Dict({
            # Board state (rank-based, same as single-leg)
            "ranked_positions": spaces.Box(
                low=0, high=20,
                shape=(NUM_CAMELS,),
                dtype=np.int8,
            ),
            "dice_rolled_by_rank": spaces.MultiBinary(NUM_CAMELS),
            "leg_tiles_available_by_rank": spaces.Box(
                low=0, high=5,
                shape=(NUM_CAMELS,),
                dtype=np.int8,
            ),
            
            # Game progress
            "current_leg": spaces.Box(low=1, high=20, shape=(1,), dtype=np.int8),
            "game_progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            
            # Desert tiles (16 spaces: 0=none, 1=oasis, -1=mirage)
            "desert_tiles": spaces.Box(
                low=-1, high=1,
                shape=(NUM_SPACES,),
                dtype=np.int8,
            ),
            "my_desert_tile_placed": spaces.MultiBinary(1),
            
            # Game-end bet tracking (total bets per camel, by rank)
            # Game-end bet tracking (total bets placed only)
            "game_winner_bet_counts": spaces.Box(
                low=0, high=40,  # Max 5 bets per 8 players = 40
                shape=(1,),
                dtype=np.int8,
            ),
            "game_loser_bet_counts": spaces.Box(
                low=0, high=40,
                shape=(1,),
                dtype=np.int8,
            ),
            # Which camels agent has bet on (winner bets + loser bets)
            "my_game_bets_mask": spaces.MultiBinary(NUM_CAMELS * 2),
            
            # Opponent awareness
            "num_players": spaces.Box(low=2, high=8, shape=(1,), dtype=np.int8),
            "my_rank": spaces.Box(low=1, high=8, shape=(1,), dtype=np.int8),
            "leader_coin_gap": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int16),
        })
        
        # Action space: 18 actions
        # 0: Roll
        # 1-5: Leg bet on ranked camel
        # 6: Place oasis, 7: Place mirage
        # 8-12: Game winner bet, 13-17: Game loser bet
        self.action_space = spaces.Discrete(NUM_FULL_GAME_ACTIONS)
        
        # Game state
        self.game_state: Optional[GameState] = None
        
        # Tracking
        self._current_player = 0
    
    def _create_opponents(self) -> None:
        """Create opponent instances."""
        self._opponents = []
        for _ in range(self.num_opponents):
            if self.opponent_type == "mc":
                self._opponents.append(
                    MCOpponent(
                        simulations=self.opponent_simulations,
                        seed=self._rng.integers(0, 2**31)
                    )
                )
            elif self.opponent_type == "random":
                self._opponents.append(
                    RandomOpponent(seed=self._rng.integers(0, 2**31))
                )
            else:  # mixed
                if self._rng.random() < 0.5:
                    self._opponents.append(
                        MCOpponent(
                            simulations=self.opponent_simulations,
                            seed=self._rng.integers(0, 2**31)
                        )
                    )
                else:
                    self._opponents.append(
                        RandomOpponent(seed=self._rng.integers(0, 2**31))
                    )
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset to a new game."""
        super().reset(seed=seed)
        
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Create new game state
        self.game_state = GameState.create_random_start(rng=self._rng)
        self.game_state.num_players = self.num_players
        # Re-initialize player lists for correct player count
        self.game_state.player_bets = [[] for _ in range(self.num_players)]
        self.game_state.player_coins = [0] * self.num_players
        self.game_state.game_winner_bets = []
        self.game_state.game_loser_bets = []
        
        # Randomize agent position
        self.agent_index = self._rng.integers(0, self.num_players)
        self._current_player = 0
        
        # Play opponent actions until it's agent's turn
        while self._current_player != self.agent_index:
            self._play_opponent_turn(self._current_player)
            self._current_player = (self._current_player + 1) % self.num_players
            
            if self.game_state.is_leg_complete():
                self._end_leg()
                self._current_player = 0
        
        if self.render_mode == "human":
            print(f"\n=== NEW GAME (Agent is Player {self.agent_index}) ===")
            print(self.game_state)
        
        return self._get_observation(), self._get_info()
    
    def step(
        self, 
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute agent action and opponent turns."""
        assert self.game_state is not None, "Must call reset() first"
        
        # Check valid action
        valid_mask = self.action_masks()
        if not valid_mask[action]:
            if self.render_mode == "human":
                print(f"  [INVALID] Agent tried action {action}")
            return (
                self._get_observation(),
                float(INVALID_ACTION_PENALTY),
                False,
                False,
                self._get_info(),
            )
        
        # Execute agent action
        self._execute_full_game_action(self.agent_index, action)
        if self.render_mode == "human":
            self._render_action(self.agent_index, action)
        
        # Check for game end or leg end
        if self.game_state.is_game_complete():
            return self._finalize_game()
        
        if self.game_state.is_leg_complete():
            self._end_leg()
        
        # Play opponent turns until back to agent
        self._current_player = (self.agent_index + 1) % self.num_players
        while self._current_player != self.agent_index:
            self._play_opponent_turn(self._current_player)
            
            if self.game_state.is_game_complete():
                return self._finalize_game()
            
            if self.game_state.is_leg_complete():
                self._end_leg()
            
            self._current_player = (self._current_player + 1) % self.num_players
        
        return self._get_observation(), 0.0, False, False, self._get_info()
    
    def _play_opponent_turn(self, player: int) -> None:
        """Execute one opponent's turn."""
        # Map player index to opponent index
        if player < self.agent_index:
            opp_idx = player
        else:
            opp_idx = player - 1
        
        if opp_idx >= len(self._opponents):
            return
        
        opponent = self._opponents[opp_idx]
        
        # Get valid actions for this player
        valid_actions = self.game_state.get_valid_actions(player)
        if not valid_actions:
            return
        
        # Opponents use leg-bet logic (actions 0-5)
        action = opponent.get_action(self.game_state, player)
        
        try:
            execute_action(self.game_state, player, action, self._rng)
            if self.render_mode == "human":
                self._render_action(player, action)
        except ValueError:
            pass
    
    def _execute_full_game_action(self, player: int, action: int) -> None:
        """Execute a full-game action."""
        if action == ACTION_ROLL:
            self.game_state.roll_die_for_player(player, self._rng)
        
        elif 1 <= action <= 5:
            # Leg bet (rank-based)
            rankings = self.game_state.get_rankings()
            rank_idx = BET_ACTION_TO_RANK[action]
            camel = rankings[rank_idx]
            self.game_state.take_bet_tile(player, camel)
        
        elif action == ACTION_PLACE_OASIS:
            valid_spaces = self.game_state.can_place_desert_tile(player)
            if valid_spaces:
                # Choose optimal space (in front of leader)
                leader_pos = self.game_state.get_leader_position()
                best = min(valid_spaces, key=lambda s: abs(s - leader_pos - 2))
                self.game_state.place_desert_tile(player, best, is_oasis=True)
        
        elif action == ACTION_PLACE_MIRAGE:
            valid_spaces = self.game_state.can_place_desert_tile(player)
            if valid_spaces:
                # Choose optimal space (in front of leader)
                leader_pos = self.game_state.get_leader_position()
                best = min(valid_spaces, key=lambda s: abs(s - leader_pos - 2))
                self.game_state.place_desert_tile(player, best, is_oasis=False)
        
        elif ACTION_GAME_WINNER_1ST <= action <= ACTION_GAME_WINNER_1ST + 4:
            # Game winner bet
            rank_idx = GAME_WINNER_ACTION_TO_RANK[action]
            rankings = self.game_state.get_rankings()
            camel = rankings[rank_idx]
            self.game_state.place_game_winner_bet(player, camel)
        
        elif ACTION_GAME_LOSER_1ST <= action <= ACTION_GAME_LOSER_1ST + 4:
            # Game loser bet
            rank_idx = GAME_LOSER_ACTION_TO_RANK[action]
            rankings = self.game_state.get_rankings()
            camel = rankings[rank_idx]
            self.game_state.place_game_loser_bet(player, camel)
    
    def _end_leg(self) -> None:
        """End current leg and start new one."""
        # Score leg bets
        self.game_state.score_bets()
        
        if self.render_mode == "human":
            print(f"\n=== LEG {self.game_state.current_leg} COMPLETE ===")
            print(f"  Rankings: {[CAMEL_NAMES[c] for c in self.game_state.get_rankings()]}")
        
        # Start new leg
        self.game_state.start_new_leg()
    
    def _finalize_game(self) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Finalize game and calculate reward."""
        # Score final leg if needed
        if not self.game_state.is_leg_complete():
            self.game_state.score_bets()
        
        # Score game-end bets
        self.game_state.score_game_end_bets()
        
        if self.render_mode == "human":
            print(f"\n=== GAME COMPLETE ===")
            print(f"  Final rankings: {[CAMEL_NAMES[c] for c in self.game_state.get_rankings()]}")
            print(f"  Final coins: {self.game_state.player_coins}")
        
        # Calculate reward
        reward = self._calculate_reward()
        
        return self._get_observation(), reward, True, False, self._get_info()
    
    def _calculate_reward(self) -> float:
        """Calculate final reward based on reward mode."""
        my_coins = self.game_state.player_coins[self.agent_index]
        max_coins = max(self.game_state.player_coins)
        
        if self.reward_mode == "win":
            return 1.0 if my_coins == max_coins else -1.0
        
        elif self.reward_mode == "coins":
            return float(my_coins)
        
        else:  # combined
            won = 1.0 if my_coins == max_coins else 0.0
            normalized = my_coins / 50.0
            return 0.7 * won + 0.3 * normalized
    
    def action_masks(self) -> np.ndarray:
        """Return boolean mask of valid actions."""
        masks = np.zeros(NUM_FULL_GAME_ACTIONS, dtype=bool)
        
        if self.game_state is None:
            return masks
        
        rankings = self.game_state.get_rankings()
        
        # Roll is valid if dice remain
        if self.game_state.dice_remaining:
            masks[ACTION_ROLL] = True
        
        # Leg bets (1-5)
        for action, rank_idx in BET_ACTION_TO_RANK.items():
            camel = rankings[rank_idx]
            if self.game_state.tiles_remaining[camel]:
                masks[action] = True
        
        # Desert tiles (6-7)
        valid_spaces = self.game_state.can_place_desert_tile(self.agent_index)
        if valid_spaces:
            masks[ACTION_PLACE_OASIS] = True
            masks[ACTION_PLACE_MIRAGE] = True
        
        # Game winner bets (8-12)
        can_win = self.game_state.can_bet_game_winner(self.agent_index)
        for action, rank_idx in GAME_WINNER_ACTION_TO_RANK.items():
            camel = rankings[rank_idx]
            if camel in can_win:
                masks[action] = True
        
        # Game loser bets (13-17)
        can_lose = self.game_state.can_bet_game_loser(self.agent_index)
        for action, rank_idx in GAME_LOSER_ACTION_TO_RANK.items():
            camel = rankings[rank_idx]
            if camel in can_lose:
                masks[action] = True
        
        return masks
    
    def _get_observation(self) -> dict[str, np.ndarray]:
        """Build observation dictionary."""
        state = self.game_state
        rankings = state.get_rankings()
        
        # Ranked positions
        ranked_positions = np.array([
            state.get_camel_position(camel)[0]
            for camel in rankings
        ], dtype=np.int8)
        
        # Dice status by rank
        dice_rolled = np.array([
            0 if camel in state.dice_remaining else 1
            for camel in rankings
        ], dtype=np.int8)
        
        # Tiles by rank
        tiles = np.array([
            state.get_top_tile(camel)
            for camel in rankings
        ], dtype=np.int8)
        
        # Game progress
        leader_pos = state.get_leader_position()
        game_progress = np.array([leader_pos / 15.0], dtype=np.float32)
        
        # Desert tiles array
        desert_arr = np.zeros(NUM_SPACES, dtype=np.int8)
        for space, (owner, effect) in state.desert_tiles.items():
            desert_arr[space] = effect
        
        # My desert tile placed?
        my_tile_placed = any(
            o == self.agent_index for s, (o, e) in state.desert_tiles.items()
        )
        
        # Game-end bet counts (total only)
        total_winner_bets = len(state.game_winner_bets)
        total_loser_bets = len(state.game_loser_bets)
        
        # My game bets mask (5 winner + 5 loser)
        my_bets = np.zeros(NUM_CAMELS * 2, dtype=np.int8)
        my_winner_bets = {c for p, c in state.game_winner_bets if p == self.agent_index}
        my_loser_bets = {c for p, c in state.game_loser_bets if p == self.agent_index}
        
        for i, camel in enumerate(rankings):
            if camel in my_winner_bets:
                my_bets[i] = 1
            if camel in my_loser_bets:
                my_bets[NUM_CAMELS + i] = 1
        
        # My rank in coins
        my_coins = state.player_coins[self.agent_index]
        sorted_coins = sorted(state.player_coins, reverse=True)
        my_rank = sorted_coins.index(my_coins) + 1
        
        # Leader gap
        leader_coins = max(state.player_coins)
        leader_gap = max(0, leader_coins - my_coins)
        
        return {
            "ranked_positions": ranked_positions,
            "dice_rolled_by_rank": dice_rolled,
            "leg_tiles_available_by_rank": tiles,
            "current_leg": np.array([state.current_leg], dtype=np.int8),
            "game_progress": game_progress,
            "desert_tiles": desert_arr,
            "my_desert_tile_placed": np.array([int(my_tile_placed)], dtype=np.int8),
            "game_winner_bet_counts": np.array([total_winner_bets], dtype=np.int8),
            "game_loser_bet_counts": np.array([total_loser_bets], dtype=np.int8),
            "my_game_bets_mask": my_bets,
            "num_players": np.array([self.num_players], dtype=np.int8),
            "my_rank": np.array([my_rank], dtype=np.int8),
            "leader_coin_gap": np.array([leader_gap], dtype=np.int16),
        }
    
    def _get_info(self) -> dict[str, Any]:
        """Return auxiliary info."""
        if self.game_state is None:
            return {}
        
        return {
            "rankings": self.game_state.get_rankings(),
            "agent_index": self.agent_index,
            "current_leg": self.game_state.current_leg,
            "player_coins": self.game_state.player_coins,
        }
    
    def _render_action(self, player: int, action: int) -> None:
        """Print action taken."""
        player_name = "Agent" if player == self.agent_index else f"Bot {player}"
        
        if action == ACTION_ROLL:
            print(f"  {player_name}: Roll")
        elif 1 <= action <= 5:
            rank_idx = BET_ACTION_TO_RANK[action]
            camel = self.game_state.get_rankings()[rank_idx]
            print(f"  {player_name}: Leg bet on {CAMEL_NAMES[camel]}")
        elif action == ACTION_PLACE_OASIS:
            print(f"  {player_name}: Place oasis")
        elif action == ACTION_PLACE_MIRAGE:
            print(f"  {player_name}: Place mirage")
        elif ACTION_GAME_WINNER_1ST <= action <= ACTION_GAME_WINNER_1ST + 4:
            rank_idx = GAME_WINNER_ACTION_TO_RANK[action]
            camel = self.game_state.get_rankings()[rank_idx]
            print(f"  {player_name}: Game winner bet on {CAMEL_NAMES[camel]}")
        elif ACTION_GAME_LOSER_1ST <= action <= ACTION_GAME_LOSER_1ST + 4:
            rank_idx = GAME_LOSER_ACTION_TO_RANK[action]
            camel = self.game_state.get_rankings()[rank_idx]
            print(f"  {player_name}: Game loser bet on {CAMEL_NAMES[camel]}")
    
    def render(self) -> None:
        """Render current state."""
        if self.render_mode == "human" and self.game_state is not None:
            print(f"\n=== Current State (Leg {self.game_state.current_leg}) ===")
            print(self.game_state)
    
    def close(self) -> None:
        """Clean up."""
        pass
