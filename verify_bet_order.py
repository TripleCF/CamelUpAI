
import sys
import os

# Ensure we can import camel_leg
sys.path.append(os.getcwd())

from camel_leg.game_state import GameState
from camel_leg.constants import BLUE, FINISH_LINE

def  verify_game_end_bet_order():
    print("=== Verifying Game End Bet Order ===")
    
    # 1. Initialize GameState
    state = GameState()
    # Ensure at least 2 players
    state.num_players = 2
    state.player_coins = [0, 0]
    
    # Place all camels on board (Spaces 0, 1, 2)
    # This prevents "Camel not found" error
    # Blue: Space 0, Green: Space 0, Orange: Space 1, Yellow: Space 1, White: Space 2
    state.board = [
        [0, 1],   # Space 1
        [2, 3],   # Space 2
        [4],      # Space 3
        [], [], [], [], [], [], [], [], [], [], [], [], [] # Spaces 4-16
    ]
    # Ensure board length is 16
    assert len(state.board) == 16
    
    # 2. Player 1 bets on Blue (First!)
    print("Player 1 betting on Blue...")
    state.place_game_winner_bet(player=1, camel=BLUE)
    
    # 3. Player 0 bets on Blue (Second!)
    print("Player 0 betting on Blue...")
    state.place_game_winner_bet(player=0, camel=BLUE)
    
    # Verify internal state structure
    print(f"Game Winner Bets: {state.game_winner_bets}")
    assert state.game_winner_bets == [(1, BLUE), (0, BLUE)], \
        f"Expected [(1, 0), (0, 0)], got {state.game_winner_bets}"
    
    # 4. Force Blue to win
    # Remove Blue from Space 1 (index 0)
    if BLUE in state.board[0]:
        state.board[0].remove(BLUE)
    
    # Place Blue at Space 16
    state.board[FINISH_LINE - 1].append(BLUE)
    # Ensure no other camels are ahead (others at start)
    
    # 5. Score Game End Bets
    print("Scoring game end bets...")
    state.score_game_end_bets()
    
    print(f"Player Coins: {state.player_coins}")
    
    # 6. Verify Payouts
    # Payouts are (8, 5, 3, 2...)
    # Player 1 (First bettor) should get 8
    # Player 0 (Second bettor) should get 5
    
    p1_coins = state.player_coins[1]
    p0_coins = state.player_coins[0]
    
    if p1_coins == 8 and p0_coins == 5:
        print("✅ SUCCESS: Player 1 (first bettor) got 8, Player 0 got 5.")
    else:
        print(f"❌ FAILURE: Expected P1=8, P0=5. Got P1={p1_coins}, P0={p0_coins}")
        exit(1)

if __name__ == "__main__":
    verify_game_end_bet_order()
