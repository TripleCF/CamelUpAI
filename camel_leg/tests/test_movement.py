"""
Unit tests for camel movement and stacking logic.
"""

import pytest
import numpy as np

from camel_leg.constants import BLUE, GREEN, ORANGE, YELLOW, WHITE, CAMEL_NAMES
from camel_leg.game_state import GameState


class TestSingleCamelMove:
    """Tests for basic single camel movement."""
    
    def test_single_camel_move_to_empty_space(self):
        """Camel moves to an empty space."""
        state = GameState()
        state.board[0] = [BLUE]  # Blue on space 1
        
        state.move_camel(BLUE, 2)
        
        assert state.board[0] == []  # Original space empty
        assert state.board[2] == [BLUE]  # Blue now on space 3
    
    def test_single_camel_move_maximum_distance(self):
        """Camel moves maximum distance (3)."""
        state = GameState()
        state.board[5] = [GREEN]  # Green on space 6
        
        state.move_camel(GREEN, 3)
        
        assert state.board[5] == []
        assert state.board[8] == [GREEN]  # Space 9
    
    def test_single_camel_move_to_end_of_board(self):
        """Camel on space 14 moves 3, capped at space 16."""
        state = GameState()
        state.board[13] = [ORANGE]  # Space 14 (0-indexed: 13)
        
        state.move_camel(ORANGE, 3)
        
        # Should be capped at space 16 (index 15), not 17
        assert state.board[13] == []
        assert state.board[15] == [ORANGE]


class TestCamelLandsOnStack:
    """Tests for camel landing on occupied space."""
    
    def test_camel_lands_on_stack_goes_on_top(self):
        """Camel landing on occupied space goes on top of stack."""
        state = GameState()
        state.board[3] = [BLUE]  # Blue on space 4
        state.board[0] = [GREEN]  # Green on space 1
        
        state.move_camel(GREEN, 3)  # Green moves 3 to space 4
        
        assert state.board[0] == []
        assert state.board[3] == [BLUE, GREEN]  # Green on top of Blue
    
    def test_camel_lands_on_multi_stack(self):
        """Camel lands on a stack with multiple camels."""
        state = GameState()
        state.board[5] = [BLUE, GREEN, ORANGE]  # Stack of 3 on space 6
        state.board[2] = [YELLOW]  # Yellow on space 3
        
        state.move_camel(YELLOW, 3)  # Yellow moves to space 6
        
        assert state.board[2] == []
        assert state.board[5] == [BLUE, GREEN, ORANGE, YELLOW]  # Yellow on top


class TestMiddleCamelMoves:
    """Tests for middle camel carrying camels above."""
    
    def test_middle_camel_carries_top_leaves_bottom(self):
        """Middle camel carries camels above, leaves camels below."""
        state = GameState()
        state.board[2] = [BLUE, GREEN, ORANGE]  # Stack: Blue (bottom), Green, Orange (top)
        
        state.move_camel(GREEN, 2)  # Green moves, carries Orange
        
        assert state.board[2] == [BLUE]  # Blue stays
        assert state.board[4] == [GREEN, ORANGE]  # Green and Orange moved
    
    def test_bottom_camel_moves_carries_all(self):
        """Bottom camel moves, carries entire stack."""
        state = GameState()
        state.board[0] = [BLUE, GREEN, ORANGE, YELLOW, WHITE]  # All 5 stacked
        
        state.move_camel(BLUE, 1)  # Bottom moves
        
        assert state.board[0] == []
        assert state.board[1] == [BLUE, GREEN, ORANGE, YELLOW, WHITE]
    
    def test_top_camel_moves_alone(self):
        """Top camel moves, carries nothing (except itself)."""
        state = GameState()
        state.board[3] = [BLUE, GREEN, ORANGE]
        
        state.move_camel(ORANGE, 2)  # Top camel moves
        
        assert state.board[3] == [BLUE, GREEN]  # Others stay
        assert state.board[5] == [ORANGE]  # Orange moves alone


class TestStackLandsOnStack:
    """Tests for partial stack landing on another stack."""
    
    def test_stack_lands_on_stack_order_preserved(self):
        """Moving stack lands on another stack, order preserved."""
        state = GameState()
        state.board[0] = [BLUE, GREEN]  # Blue, Green on space 1
        state.board[2] = [ORANGE, YELLOW]  # Orange, Yellow on space 3
        
        state.move_camel(BLUE, 2)  # Blue moves, carries Green
        
        assert state.board[0] == []
        assert state.board[2] == [ORANGE, YELLOW, BLUE, GREEN]
        # Order: Orange (bottom), Yellow, Blue, Green (top)
    
    def test_partial_stack_lands_on_stack(self):
        """Middle of one stack lands on another stack."""
        state = GameState()
        state.board[0] = [BLUE, GREEN, ORANGE]  # 3 camels
        state.board[3] = [YELLOW, WHITE]  # 2 camels
        
        state.move_camel(GREEN, 3)  # Green carries Orange, lands on Yellow/White
        
        assert state.board[0] == [BLUE]  # Blue stays
        assert state.board[3] == [YELLOW, WHITE, GREEN, ORANGE]


class TestGetCamelPosition:
    """Tests for position lookup."""
    
    def test_get_position_single_camel(self):
        """Get position of single camel on space."""
        state = GameState()
        state.board[5] = [BLUE]
        
        space, height = state.get_camel_position(BLUE)
        
        assert space == 5
        assert height == 0  # Bottom of stack
    
    def test_get_position_in_stack(self):
        """Get position of camel in middle of stack."""
        state = GameState()
        state.board[3] = [BLUE, GREEN, ORANGE]
        
        assert state.get_camel_position(BLUE) == (3, 0)
        assert state.get_camel_position(GREEN) == (3, 1)
        assert state.get_camel_position(ORANGE) == (3, 2)
    
    def test_get_position_camel_not_found(self):
        """Raise error if camel not on board."""
        state = GameState()
        state.board[0] = [BLUE]
        
        with pytest.raises(ValueError, match="not found"):
            state.get_camel_position(GREEN)
