"""
Unit tests for betting mechanics.
"""

import pytest

from camel_leg.constants import BLUE, GREEN, ORANGE, YELLOW, WHITE, CAMEL_NAMES
from camel_leg.game_state import GameState


class TestBetTileTaking:
    """Tests for taking bet tiles."""
    
    def test_take_first_tile_returns_5(self):
        """First tile taken from a camel is worth 5."""
        state = GameState()
        
        tile_value = state.take_bet_tile(player=0, camel=BLUE)
        
        assert tile_value == 5
    
    def test_take_tiles_in_order(self):
        """Tiles are taken in order: 5, 3, 2, 1."""
        state = GameState()
        
        assert state.take_bet_tile(0, BLUE) == 5
        assert state.take_bet_tile(1, BLUE) == 3
        assert state.take_bet_tile(2, BLUE) == 2
        assert state.take_bet_tile(3, BLUE) == 1
    
    def test_take_tile_updates_remaining(self):
        """Taking tile updates tiles_remaining correctly."""
        state = GameState()
        
        assert state.tiles_remaining[BLUE] == [5, 3, 2, 1]
        state.take_bet_tile(0, BLUE)
        assert state.tiles_remaining[BLUE] == [3, 2, 1]
    
    def test_take_tile_records_bet(self):
        """Taking tile records the bet for the player."""
        state = GameState()
        
        state.take_bet_tile(player=2, camel=GREEN)
        
        assert state.player_bets[2] == [(GREEN, 5)]
    
    def test_multiple_bets_same_player(self):
        """Player can take multiple tiles for different camels."""
        state = GameState()
        
        state.take_bet_tile(0, BLUE)
        state.take_bet_tile(0, GREEN)
        state.take_bet_tile(0, ORANGE)
        
        assert len(state.player_bets[0]) == 3
        assert state.player_bets[0][0] == (BLUE, 5)
        assert state.player_bets[0][1] == (GREEN, 5)
        assert state.player_bets[0][2] == (ORANGE, 5)


class TestBetTileExhaustion:
    """Tests for exhausted bet tiles."""
    
    def test_no_tiles_remaining_raises_error(self):
        """Taking tile when none remain raises ValueError."""
        state = GameState()
        
        # Take all 4 tiles
        for player in range(4):
            state.take_bet_tile(player, BLUE)
        
        # 5th attempt should fail
        with pytest.raises(ValueError, match="No tiles remaining"):
            state.take_bet_tile(0, BLUE)
    
    def test_get_top_tile_returns_zero_when_empty(self):
        """get_top_tile returns 0 when no tiles remain."""
        state = GameState()
        
        # Take all tiles
        for player in range(4):
            state.take_bet_tile(player, GREEN)
        
        assert state.get_top_tile(GREEN) == 0


class TestBetScoring:
    """Tests for scoring bets at end of leg."""
    
    def test_first_place_bet_pays_tile_value(self):
        """Bet on 1st place camel pays tile value."""
        state = GameState()
        state.board[10] = [BLUE]  # Blue in 1st
        state.board[5] = [GREEN]
        state.board[0] = [ORANGE, YELLOW, WHITE]
        
        state.take_bet_tile(0, BLUE)  # Player 0 bets Blue@5
        
        deltas = state.score_bets()
        
        assert deltas[0] == 5  # +5 for correct 1st place
    
    def test_second_place_bet_pays_one(self):
        """Bet on 2nd place camel pays +1."""
        state = GameState()
        state.board[10] = [BLUE]
        state.board[5] = [GREEN]  # Green in 2nd
        state.board[0] = [ORANGE, YELLOW, WHITE]
        
        state.take_bet_tile(0, GREEN)  # Bet on 2nd place
        
        deltas = state.score_bets()
        
        assert deltas[0] == 1
    
    def test_wrong_bet_loses_one(self):
        """Bet on camel not in 1st or 2nd loses -1."""
        state = GameState()
        state.board[10] = [BLUE]
        state.board[5] = [GREEN]
        state.board[0] = [ORANGE]  # Orange in 3rd
        state.board[1] = [YELLOW]
        state.board[2] = [WHITE]
        
        state.take_bet_tile(0, ORANGE)  # Bet on 3rd place
        
        deltas = state.score_bets()
        
        assert deltas[0] == -1
    
    def test_multiple_bets_summed(self):
        """Multiple bets are summed correctly."""
        state = GameState()
        state.board[10] = [BLUE]   # 1st
        state.board[5] = [GREEN]   # 2nd
        state.board[0] = [ORANGE, YELLOW, WHITE]  # 3rd, 4th, 5th
        
        state.take_bet_tile(0, BLUE)    # +5
        state.take_bet_tile(0, GREEN)   # +1
        state.take_bet_tile(0, ORANGE)  # -1
        
        deltas = state.score_bets()
        
        assert deltas[0] == 5 + 1 - 1  # = 5
    
    def test_scoring_updates_player_coins(self):
        """Scoring adds deltas to player coins."""
        state = GameState()
        state.player_coins[0] = 10
        state.board[10] = [BLUE]
        state.board[5] = [GREEN]
        state.board[0] = [ORANGE, YELLOW, WHITE]
        
        state.take_bet_tile(0, BLUE)  # +5
        state.score_bets()
        
        assert state.player_coins[0] == 15


class TestValidActions:
    """Tests for get_valid_actions."""
    
    def test_roll_valid_when_dice_remain(self):
        """Roll action is valid when dice remain."""
        state = GameState()
        
        valid = state.get_valid_actions(0)
        
        assert 0 in valid  # ACTION_ROLL = 0
    
    def test_roll_invalid_when_no_dice(self):
        """Roll action invalid when all dice used."""
        state = GameState()
        state.dice_remaining = set()  # No dice left
        
        valid = state.get_valid_actions(0)
        
        assert 0 not in valid
    
    def test_bet_valid_when_tiles_remain(self):
        """Bet action valid when tiles remain for camel."""
        state = GameState()
        
        valid = state.get_valid_actions(0)
        
        # Actions 1-5 are bets on camels 0-4
        assert all(a in valid for a in [1, 2, 3, 4, 5])
    
    def test_bet_invalid_when_no_tiles(self):
        """Bet action invalid when no tiles for that camel."""
        state = GameState()
        state.tiles_remaining[BLUE] = []  # No Blue tiles
        
        valid = state.get_valid_actions(0)
        
        assert 1 not in valid  # ACTION_BET_BLUE = 1
        assert 2 in valid  # Other camels still valid
