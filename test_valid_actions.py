
import numpy as np
from camel_leg.game_state import GameState
from camel_leg.constants import NUM_PLAYERS

# Mock a full game state by adding required attributes if they aren't there
# but GameState usually has them if created via create_random_start in certain modes.
# Actually let's just create one and see.

state = GameState.create_random_start()
# GameState attributes for full game:
state.game_winner_bets = []
state.game_loser_bets = []
state.desert_tiles = {}

print("Valid actions for Player 0:", state.get_valid_actions(0))
