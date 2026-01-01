"""
Unit tests for camel ranking logic.
"""

import pytest

from camel_leg.constants import BLUE, GREEN, ORANGE, YELLOW, WHITE, CAMEL_NAMES
from camel_leg.game_state import GameState


class TestRankingByPosition:
    """Tests for ranking based on track position."""
    
    def test_ranking_by_position_simple(self):
        """Camels ranked by position, furthest first."""
        state = GameState()
        state.board[10] = [BLUE]   # Space 11
        state.board[5] = [GREEN]   # Space 6
        state.board[3] = [ORANGE]  # Space 4
        state.board[1] = [YELLOW]  # Space 2
        state.board[0] = [WHITE]   # Space 1
        
        rankings = state.get_rankings()
        
        assert rankings == [BLUE, GREEN, ORANGE, YELLOW, WHITE]
    
    def test_ranking_spread_across_board(self):
        """Ranking with camels spread across board."""
        state = GameState()
        state.board[15] = [WHITE]  # Space 16 (furthest)
        state.board[8] = [ORANGE]
        state.board[5] = [YELLOW]
        state.board[2] = [BLUE]
        state.board[0] = [GREEN]   # Space 1 (last)
        
        rankings = state.get_rankings()
        
        assert rankings[0] == WHITE  # 1st
        assert rankings[-1] == GREEN  # 5th


class TestRankingByStackHeight:
    """Tests for ranking tiebreak by stack height."""
    
    def test_ranking_same_space_by_height(self):
        """Same space: higher in stack = better rank."""
        state = GameState()
        state.board[5] = [BLUE, GREEN, ORANGE]  # Blue bottom, Orange top
        state.board[0] = [YELLOW]
        state.board[1] = [WHITE]
        
        rankings = state.get_rankings()
        
        # Orange (top at space 6) beats Green beats Blue
        assert rankings[0] == ORANGE
        assert rankings[1] == GREEN
        assert rankings[2] == BLUE
        # Then White (space 2) beats Yellow (space 1)
        assert rankings[3] == WHITE
        assert rankings[4] == YELLOW
    
    def test_ranking_all_stacked(self):
        """All 5 camels on same space: pure height ranking."""
        state = GameState()
        state.board[3] = [WHITE, YELLOW, ORANGE, GREEN, BLUE]  # Top to bottom: Blue on top
        
        rankings = state.get_rankings()
        
        # Top of stack is last in the list (index 4 = BLUE here)
        assert rankings == [BLUE, GREEN, ORANGE, YELLOW, WHITE]
    
    def test_ranking_two_stacks(self):
        """Two separate stacks, position beats height."""
        state = GameState()
        state.board[10] = [BLUE, GREEN]  # Space 11
        state.board[5] = [ORANGE, YELLOW, WHITE]  # Space 6, but stack of 3
        
        rankings = state.get_rankings()
        
        # Space 11 beats space 6, regardless of stack height
        assert rankings[0] == GREEN  # Top at space 11
        assert rankings[1] == BLUE   # Bottom at space 11
        assert rankings[2] == WHITE  # Top at space 6
        assert rankings[3] == YELLOW
        assert rankings[4] == ORANGE  # Bottom at space 6


class TestRankingEdgeCases:
    """Edge cases for ranking."""
    
    def test_ranking_after_movement(self):
        """Rankings update correctly after movement."""
        state = GameState()
        state.board[5] = [BLUE, GREEN]
        state.board[3] = [ORANGE]
        state.board[0] = [YELLOW, WHITE]
        
        initial_rankings = state.get_rankings()
        assert initial_rankings[0] == GREEN  # Green 1st
        
        # Orange moves past everyone
        state.move_camel(ORANGE, 3)  # Space 4 → 7
        
        new_rankings = state.get_rankings()
        assert new_rankings[0] == ORANGE  # Now Orange 1st
        assert new_rankings[1] == GREEN   # Green 2nd
