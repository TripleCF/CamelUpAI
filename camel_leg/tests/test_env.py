"""
Integration tests for the Gymnasium environment.
"""

import pytest
import numpy as np

from camel_leg.env import CamelLegEnv
from camel_leg.constants import (
    BLUE, GREEN, ORANGE, YELLOW, WHITE,
    NUM_CAMELS, NUM_SPACES, NUM_PLAYERS, NUM_ACTIONS,
    ACTION_ROLL, INVALID_ACTION_PENALTY,
)


class TestEnvBasics:
    """Basic environment functionality tests."""
    
    def test_env_creation(self):
        """Environment can be created."""
        env = CamelLegEnv(opponent_type="random")
        assert env is not None
    
    def test_env_reset_returns_valid_obs(self):
        """Reset returns valid observation."""
        env = CamelLegEnv(opponent_type="random")
        obs, info = env.reset(seed=42)
        
        assert "ranked_positions" in obs
        assert "dice_rolled_by_rank" in obs
        assert "tiles_available_by_rank" in obs
        assert "player_coins" in obs
    
    def test_obs_shapes_correct(self):
        """Observation shapes match space definition."""
        env = CamelLegEnv(opponent_type="random")
        obs, _ = env.reset(seed=42)
        
        assert obs["ranked_positions"].shape == (NUM_CAMELS,)
        assert obs["dice_rolled_by_rank"].shape == (NUM_CAMELS,)
        assert obs["tiles_available_by_rank"].shape == (NUM_CAMELS,)
        assert obs["player_coins"].shape == (NUM_PLAYERS,)
    
    def test_initial_dice_all_remaining(self):
        """All 5 dice remain at start."""
        env = CamelLegEnv(opponent_type="random")
        obs, _ = env.reset(seed=42)
        
        # dice_rolled_by_rank should be all 0 (none rolled yet)
        assert sum(obs["dice_rolled_by_rank"]) == 0
    
    def test_initial_tiles_all_5(self):
        """All tile values are 5 at start."""
        env = CamelLegEnv(opponent_type="random")
        obs, _ = env.reset(seed=42)
        
        assert all(t == 5 for t in obs["tiles_available_by_rank"])


class TestActionMasking:
    """Tests for action masking."""
    
    def test_action_masks_shape(self):
        """Action masks have correct shape."""
        env = CamelLegEnv(opponent_type="random")
        env.reset(seed=42)
        
        masks = env.action_masks()
        
        assert masks.shape == (NUM_ACTIONS,)
        assert masks.dtype == bool
    
    def test_roll_masked_when_no_dice(self):
        """Roll action masked when no dice remain."""
        env = CamelLegEnv(opponent_type="random")
        env.reset(seed=42)
        
        # Remove all dice
        env.game_state.dice_remaining = set()
        
        masks = env.action_masks()
        
        assert masks[ACTION_ROLL] == False
    
    def test_bet_masked_when_no_tiles(self):
        """Bet action masked when no tiles for camel at that rank."""
        env = CamelLegEnv(opponent_type="random")
        env.reset(seed=42)
        
        # Get current leader and remove their tiles
        rankings = env.game_state.get_rankings()
        leader = rankings[0]
        env.game_state.tiles_remaining[leader] = []
        
        masks = env.action_masks()
        
        # Action 1 (bet on 1st place) should be masked
        assert masks[1] == False
        # Other ranks should still be valid
        assert masks[2] == True


class TestStepFunction:
    """Tests for step execution."""
    
    def test_step_returns_correct_tuple(self):
        """Step returns (obs, reward, term, trunc, info)."""
        env = CamelLegEnv(opponent_type="random")
        env.reset(seed=42)
        
        result = env.step(ACTION_ROLL)
        
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_invalid_action_penalty(self):
        """Invalid action returns penalty."""
        env = CamelLegEnv(opponent_type="random")
        env.reset(seed=42)
        
        # Get leader and remove their tiles to make action 1 (bet on 1st) invalid
        rankings = env.game_state.get_rankings()
        leader = rankings[0]
        env.game_state.tiles_remaining[leader] = []
        
        _, reward, _, _, _ = env.step(1)  # Try to bet on 1st place (no tiles)
        
        assert reward == float(INVALID_ACTION_PENALTY)
    
    def test_roll_gives_reward(self):
        """Rolling gives +1 reward (unless leg ends)."""
        env = CamelLegEnv(opponent_type="random")
        env.reset(seed=42)
        
        # Disable opponent rolling to control state
        env.game_state.dice_remaining = {BLUE}  # Only Blue die left
        
        _, reward, terminated, _, _ = env.step(ACTION_ROLL)
        
        # Reward should include roll bonus
        # (May also include bet scoring if leg ended)
        assert reward >= 1.0 or terminated


class TestLegCompletion:
    """Tests for leg completion."""
    
    def test_leg_ends_after_five_dice(self):
        """Episode terminates when all dice rolled."""
        env = CamelLegEnv(opponent_type="random")
        env.reset(seed=42)
        
        terminated = False
        steps = 0
        max_steps = 100
        
        while not terminated and steps < max_steps:
            action = ACTION_ROLL if env.action_masks()[ACTION_ROLL] else 1
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
        
        assert terminated, "Leg should terminate"
        assert steps <= max_steps, "Should terminate before max steps"
    
    def test_terminal_state_has_no_dice(self):
        """Terminal state has no dice remaining."""
        env = CamelLegEnv(opponent_type="random")
        env.reset(seed=42)
        
        terminated = False
        while not terminated:
            action = ACTION_ROLL if env.action_masks()[ACTION_ROLL] else 1
            obs, _, terminated, _, _ = env.step(action)
        
        # All dice should be rolled (dice_rolled_by_rank all 1)
        assert all(obs["dice_rolled_by_rank"] == 1)


class TestRandomOpponent:
    """Tests with random opponents."""
    
    def test_full_game_random_opponents(self):
        """Full game completes with random opponents."""
        env = CamelLegEnv(opponent_type="random", seed=42)
        
        for episode in range(5):
            obs, _ = env.reset()
            terminated = False
            total_reward = 0
            
            while not terminated:
                valid_actions = np.where(env.action_masks())[0]
                action = np.random.choice(valid_actions)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            
            # Game completed successfully
            assert terminated


class TestMCOpponent:
    """Tests with MC opponents (lightweight)."""
    
    def test_mc_opponent_creates(self):
        """MC opponent environment can be created."""
        env = CamelLegEnv(opponent_type="mc", opponent_simulations=10)
        assert env is not None
    
    def test_mc_opponent_step(self):
        """MC opponent can take actions."""
        env = CamelLegEnv(opponent_type="mc", opponent_simulations=10)
        env.reset(seed=42)
        
        # Just verify step doesn't crash
        obs, reward, terminated, truncated, info = env.step(ACTION_ROLL)
        
        assert obs is not None


class TestReproducibility:
    """Tests for deterministic behavior with seeds."""
    
    def test_reset_with_seed_deterministic(self):
        """Reset with same seed gives same initial state."""
        env = CamelLegEnv(opponent_type="random")
        
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        assert np.array_equal(obs1["ranked_positions"], obs2["ranked_positions"])
        assert np.array_equal(obs1["dice_rolled_by_rank"], obs2["dice_rolled_by_rank"])


class TestActionTranslation:
    """Tests for rank-to-color action translation."""
    
    def test_bet_on_1st_bets_on_leader(self):
        """Bet on 1st place correctly bets on leading camel."""
        env = CamelLegEnv(opponent_type="random")
        env.reset(seed=42)
        
        rankings = env.game_state.get_rankings()
        leader = rankings[0]
        
        # Action 1 = bet on 1st place
        env.step(1)
        
        # Verify the RL agent bet on the leader
        agent_bets = env.game_state.player_bets[0]
        bet_camels = [camel for camel, _ in agent_bets]
        assert leader in bet_camels
    
    def test_bet_on_2nd_bets_on_second_place(self):
        """Bet on 2nd place correctly bets on second-place camel."""
        env = CamelLegEnv(opponent_type="random")
        env.reset(seed=42)
        
        rankings = env.game_state.get_rankings()
        second = rankings[1]
        
        # Action 2 = bet on 2nd place
        env.step(2)
        
        # Verify the RL agent bet on second place
        agent_bets = env.game_state.player_bets[0]
        bet_camels = [camel for camel, _ in agent_bets]
        assert second in bet_camels
