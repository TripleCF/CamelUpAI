"""
Game constants for Camel Up (1st Edition).

Supports both single-leg simulation and full game.
"""

from typing import Final

# Camel indices (used throughout for identification)
BLUE: Final[int] = 0
GREEN: Final[int] = 1
ORANGE: Final[int] = 2
YELLOW: Final[int] = 3
WHITE: Final[int] = 4

# Convenience collections
CAMELS: Final[tuple[int, ...]] = (BLUE, GREEN, ORANGE, YELLOW, WHITE)
CAMEL_NAMES: Final[tuple[str, ...]] = ("Blue", "Green", "Orange", "Yellow", "White")
NUM_CAMELS: Final[int] = 5

# Board configuration
NUM_SPACES: Final[int] = 16  # Spaces 1-16 (we use 0-indexed internally: 0-15)
START_SPACES: Final[tuple[int, ...]] = (0, 1, 2)  # Spaces 1-3 (0-indexed: 0, 1, 2)
FINISH_LINE: Final[int] = 16  # Camel crossing space 16 triggers game end

# Dice configuration
DICE_VALUES: Final[tuple[int, ...]] = (1, 2, 3)

# Betting tiles: first-come-first-served values per camel
BET_TILE_VALUES: Final[tuple[int, ...]] = (5, 3, 2, 1)
NUM_TILES_PER_CAMEL: Final[int] = 4

# Rewards
ROLL_REWARD: Final[int] = 1
INVALID_ACTION_PENALTY: Final[int] = -10

# Players
NUM_PLAYERS: Final[int] = 4
RL_AGENT_INDEX: Final[int] = 0  # The RL agent is always player 0

# =============================================================================
# Single-leg action space (6 actions)
# =============================================================================
ACTION_ROLL: Final[int] = 0
ACTION_BET_1ST: Final[int] = 1  # Bet on current 1st place camel
ACTION_BET_2ND: Final[int] = 2  # Bet on current 2nd place camel
ACTION_BET_3RD: Final[int] = 3  # Bet on current 3rd place camel
ACTION_BET_4TH: Final[int] = 4  # Bet on current 4th place camel
ACTION_BET_5TH: Final[int] = 5  # Bet on current 5th place camel
NUM_ACTIONS: Final[int] = 6

# Mapping from action to rank index (0=1st, 1=2nd, etc.)
BET_ACTION_TO_RANK: Final[dict[int, int]] = {
    ACTION_BET_1ST: 0,
    ACTION_BET_2ND: 1,
    ACTION_BET_3RD: 2,
    ACTION_BET_4TH: 3,
    ACTION_BET_5TH: 4,
}



# =============================================================================
# Full game constants
# =============================================================================

# Desert tiles
OASIS_EFFECT: Final[int] = 1   # +1 space forward, landing camels go on TOP
MIRAGE_EFFECT: Final[int] = -1  # -1 space backward, landing camels go on BOTTOM

# Game end bet scoring (first correct bettor gets 8, second gets 5, etc.)
GAME_END_BET_PAYOUTS: Final[tuple[int, ...]] = (8, 5, 3, 2, 1, 1, 1, 1)
GAME_END_BET_PENALTY: Final[int] = -1

# =============================================================================
# Full game action space (18 actions)
# =============================================================================
# 0: Roll
# 1-5: Leg bet on ranked camel
# 6-7: Place oasis/mirage at optimal position
# 8-12: Game winner bet on ranked camel
# 13-17: Game loser bet on ranked camel

ACTION_PLACE_OASIS: Final[int] = 6
ACTION_PLACE_MIRAGE: Final[int] = 7

ACTION_GAME_WINNER_1ST: Final[int] = 8
ACTION_GAME_WINNER_2ND: Final[int] = 9
ACTION_GAME_WINNER_3RD: Final[int] = 10
ACTION_GAME_WINNER_4TH: Final[int] = 11
ACTION_GAME_WINNER_5TH: Final[int] = 12

ACTION_GAME_LOSER_1ST: Final[int] = 13
ACTION_GAME_LOSER_2ND: Final[int] = 14
ACTION_GAME_LOSER_3RD: Final[int] = 15
ACTION_GAME_LOSER_4TH: Final[int] = 16
ACTION_GAME_LOSER_5TH: Final[int] = 17

NUM_FULL_GAME_ACTIONS: Final[int] = 18

GAME_WINNER_ACTION_TO_RANK: Final[dict[int, int]] = {
    ACTION_GAME_WINNER_1ST: 0,
    ACTION_GAME_WINNER_2ND: 1,
    ACTION_GAME_WINNER_3RD: 2,
    ACTION_GAME_WINNER_4TH: 3,
    ACTION_GAME_WINNER_5TH: 4,
}

GAME_LOSER_ACTION_TO_RANK: Final[dict[int, int]] = {
    ACTION_GAME_LOSER_1ST: 0,
    ACTION_GAME_LOSER_2ND: 1,
    ACTION_GAME_LOSER_3RD: 2,
    ACTION_GAME_LOSER_4TH: 3,
    ACTION_GAME_LOSER_5TH: 4,
}

