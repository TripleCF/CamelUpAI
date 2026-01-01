"""
Core game state management for Camel Up.

Supports both single-leg simulation and full game with desert tiles,
game-end bets, and multi-leg gameplay.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .constants import (
    BET_TILE_VALUES,
    CAMELS,
    CAMEL_NAMES,
    DICE_VALUES,
    FINISH_LINE,
    GAME_END_BET_PAYOUTS,
    GAME_END_BET_PENALTY,
    MIRAGE_EFFECT,
    NUM_CAMELS,
    NUM_PLAYERS,
    NUM_SPACES,
    NUM_TILES_PER_CAMEL,
    OASIS_EFFECT,
    ROLL_REWARD,
    START_SPACES,
)


@dataclass
class GameState:
    """
    Manages the complete state of a Camel Up game.
    
    Supports both single-leg simulation and full multi-leg game.
    
    Attributes:
        board: List of 16 stacks (one per space). Each stack is a list of
               camel indices, ordered bottom to top. Empty spaces have [].
        dice_remaining: Set of camel indices whose dice are still in the pyramid.
        tiles_remaining: Dict mapping camel index to list of remaining tile values.
        player_bets: List of player lists containing (camel, tile_value) tuples.
        player_coins: List of integers representing each player's coin total.
        
        # Full game fields:
        desert_tiles: Dict mapping space index to (owner, effect) tuple.
                     Effect is +1 (oasis) or -1 (mirage).
        game_winner_bets: List of player lists containing camel indices bet.
        game_loser_bets: List of player lists containing camel indices bet.
        current_leg: Current leg number (starts at 1).
        game_complete: Whether the game has ended.
        num_players: Number of players (can vary 2-8).
    """
    
    # Core leg fields
    board: list[list[int]] = field(default_factory=list)
    dice_remaining: set[int] = field(default_factory=set)
    tiles_remaining: dict[int, list[int]] = field(default_factory=dict)
    player_bets: list[list[tuple[int, int]]] = field(default_factory=list)
    player_coins: list[int] = field(default_factory=list)
    
    # Full game fields
    desert_tiles: dict[int, tuple[int, int]] = field(default_factory=dict)
    game_winner_bets: list[tuple[int, int]] = field(default_factory=list)
    game_loser_bets: list[tuple[int, int]] = field(default_factory=list)
    current_leg: int = 1
    game_complete: bool = False
    num_players: int = NUM_PLAYERS
    
    def __post_init__(self) -> None:
        """Initialize empty board if not provided."""
        if not self.board:
            self.board = [[] for _ in range(NUM_SPACES)]
        if not self.dice_remaining:
            self.dice_remaining = set(CAMELS)
        if not self.tiles_remaining:
            self.tiles_remaining = {c: list(BET_TILE_VALUES) for c in CAMELS}
        if not self.player_bets:
            self.player_bets = [[] for _ in range(self.num_players)]
        if not self.player_coins:
            self.player_coins = [0] * self.num_players
        if not self.game_winner_bets:
            self.game_winner_bets = []
        if not self.game_loser_bets:
            self.game_loser_bets = []
    
    @classmethod
    def create_random_start(cls, rng: Optional[np.random.Generator] = None) -> GameState:
        """
        Create a new game state with camels randomly placed on spaces 1-3.
        
        Each camel is assigned to a random starting space (1, 2, or 3).
        Camels on the same space form a stack in random order.
        
        Args:
            rng: NumPy random generator for reproducibility. If None, uses default.
            
        Returns:
            A new GameState with random initial camel placement.
            
        Example:
            >>> state = GameState.create_random_start()
            >>> sum(len(stack) for stack in state.board)  # Total camels
            5
        """
        if rng is None:
            rng = np.random.default_rng()
        
        state = cls()
        
        # Assign each camel to a random starting space
        camel_order = list(CAMELS)
        rng.shuffle(camel_order)
        
        for camel in camel_order:
            space = rng.choice(START_SPACES)
            state.board[space].append(camel)
        
        return state
    
    def get_camel_position(self, camel: int) -> tuple[int, int]:
        """
        Get the position and stack height of a camel.
        
        Args:
            camel: Camel index (0-4).
            
        Returns:
            Tuple of (space_index, stack_height) where stack_height 0 is bottom.
            
        Raises:
            ValueError: If camel is not found on the board.
            
        Example:
            >>> state.board[5] = [1, 0, 2]  # Green, Blue, Orange stacked on space 6
            >>> state.get_camel_position(0)  # Blue
            (5, 1)  # Space 6 (0-indexed as 5), height 1 (middle)
        """
        for space_idx, stack in enumerate(self.board):
            if camel in stack:
                height = stack.index(camel)
                return (space_idx, height)
        raise ValueError(f"Camel {camel} ({CAMEL_NAMES[camel]}) not found on board")
    
    def move_camel(self, camel: int, distance: int) -> None:
        """
        Move a camel and all camels above it in the stack.
        
        Handles desert tile effects:
        - Oasis (+1): Camel moves extra space forward, lands on TOP
        - Mirage (-1): Camel moves one space backward, lands on BOTTOM
        
        The desert tile owner receives 1 coin when a camel lands on their tile.
        
        Args:
            camel: Camel index (0-4) to move.
            distance: Number of spaces to move (1, 2, or 3).
        """
        # Find current position
        current_space, height = self.get_camel_position(camel)
        
        # Get the moving group (this camel and all above it)
        moving_camels = self.board[current_space][height:]
        
        # Remove from current space
        self.board[current_space] = self.board[current_space][:height]
        
        # Calculate initial destination
        destination = current_space + distance
        
        # Check for desert tile at destination
        land_on_bottom = False
        if destination in self.desert_tiles:
            owner, effect = self.desert_tiles[destination]
            destination += effect
            # Owner gets 1 coin
            if 0 <= owner < len(self.player_coins):
                self.player_coins[owner] += 1
            # Mirage (-1) means land on bottom of stack
            if effect == MIRAGE_EFFECT:
                land_on_bottom = True
        
        # Cap destination at valid board range
        destination = max(0, min(destination, NUM_SPACES - 1))
        
        # Add to destination
        if land_on_bottom:
            # Insert at bottom of stack
            self.board[destination] = moving_camels + self.board[destination]
        else:
            # Add on top of stack (normal)
            self.board[destination].extend(moving_camels)
    
    def get_rankings(self) -> list[int]:
        """
        Get camels in rank order from 1st to 5th place.
        
        Ranking rules:
        - Primary: Furthest position on track (higher space index = better)
        - Secondary: Higher in stack (top of stack = better)
        
        Returns:
            List of camel indices in order [1st, 2nd, 3rd, 4th, 5th].
            
        Example:
            >>> state.board[5] = [0, 1]  # Blue bottom, Green top on space 6
            >>> state.board[3] = [2]  # Orange on space 4
            >>> state.get_rankings()
            [1, 0, 2, ...]  # Green 1st (top at space 6), Blue 2nd, Orange 3rd
        """
        # Build list of (camel, space, height)
        camel_info: list[tuple[int, int, int]] = []
        for camel in CAMELS:
            space, height = self.get_camel_position(camel)
            camel_info.append((camel, space, height))
        
        # Sort by space (descending), then height (descending)
        camel_info.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        return [camel for camel, _, _ in camel_info]
    
    def get_top_tile(self, camel: int) -> int:
        """
        Get the top available bet tile value for a camel.
        
        Args:
            camel: Camel index (0-4).
            
        Returns:
            The next available tile value (5, 3, 2, or 1), or 0 if no tiles remain.
        """
        tiles = self.tiles_remaining[camel]
        return tiles[0] if tiles else 0
    
    def take_bet_tile(self, player: int, camel: int) -> int:
        """
        Player takes the top bet tile for a camel.
        
        Args:
            player: Player index (0-3).
            camel: Camel index (0-4).
            
        Returns:
            The tile value taken (5, 3, 2, or 1).
            
        Raises:
            ValueError: If no tiles remain for this camel.
            
        Example:
            >>> state.tiles_remaining[0]  # Blue tiles
            [5, 3, 2, 1]
            >>> state.take_bet_tile(0, 0)  # Player 0 takes Blue tile
            5
            >>> state.tiles_remaining[0]
            [3, 2, 1]
            >>> state.player_bets[0]
            [(0, 5)]  # Player 0 has bet on Blue with value 5
        """
        if not self.tiles_remaining[camel]:
            raise ValueError(f"No tiles remaining for camel {CAMEL_NAMES[camel]}")
        
        tile_value = self.tiles_remaining[camel].pop(0)
        self.player_bets[player].append((camel, tile_value))
        return tile_value
    
    def roll_die(self, rng: Optional[np.random.Generator] = None) -> tuple[int, int]:
        """
        Draw a random die from the pyramid and roll it.
        
        A random camel's die is selected from those remaining in the pyramid.
        The die is rolled (1, 2, or 3), and the camel is moved.
        The rolling player receives ROLL_REWARD coins.
        
        Args:
            rng: NumPy random generator for reproducibility.
            
        Returns:
            Tuple of (camel_index, distance_rolled).
            
        Raises:
            ValueError: If no dice remain in the pyramid.
        """
        if not self.dice_remaining:
            raise ValueError("No dice remaining in pyramid")
        
        if rng is None:
            rng = np.random.default_rng()
        
        # Select random die from remaining
        camel = rng.choice(list(self.dice_remaining))
        self.dice_remaining.remove(camel)
        
        # Roll the die
        distance = rng.choice(DICE_VALUES)
        
        # Move the camel
        self.move_camel(camel, distance)
        
        return (camel, distance)
    
    def roll_die_for_player(
        self, 
        player: int, 
        rng: Optional[np.random.Generator] = None
    ) -> tuple[int, int]:
        """
        Player rolls a die and receives the roll reward.
        
        Args:
            player: Player index (0-3).
            rng: Random generator.
            
        Returns:
            Tuple of (camel_index, distance_rolled).
        """
        camel, distance = self.roll_die(rng)
        self.player_coins[player] += ROLL_REWARD
        return (camel, distance)
    
    def is_leg_complete(self) -> bool:
        """Check if the leg is complete (all 5 dice rolled)."""
        return len(self.dice_remaining) == 0
    
    def score_bets(self) -> dict[int, int]:
        """
        Score all player bets at the end of the leg.
        
        Scoring rules per bet tile:
        - Bet matches 1st place: +tile_value coins
        - Bet matches 2nd place: +1 coin
        - Otherwise: -1 coin
        
        Returns:
            Dict mapping player index to coins earned from bets this leg.
            
        Example:
            >>> state.player_bets[0] = [(0, 5), (1, 3)]  # Bet Blue@5, Green@3
            >>> rankings = state.get_rankings()  # [0, 2, 1, ...]  Blue 1st, Orange 2nd, Green 3rd
            >>> deltas = state.score_bets()
            >>> deltas[0]  # Player 0: +5 for Blue 1st, -1 for Green 3rd
            4
        """
        rankings = self.get_rankings()
        first_place = rankings[0]
        second_place = rankings[1]
        
        deltas: dict[int, int] = {p: 0 for p in range(self.num_players)}
        
        for player in range(self.num_players):
            for camel, tile_value in self.player_bets[player]:
                if camel == first_place:
                    deltas[player] += tile_value
                elif camel == second_place:
                    deltas[player] += 1
                else:
                    deltas[player] -= 1
        
        # Add deltas to player coins
        for player, delta in deltas.items():
            self.player_coins[player] += delta
        
        return deltas
    
    def get_valid_actions(self, player: int) -> list[int]:
        """
        Get list of valid action indices for a player.
        
        Supports both leg-only and full-game actions.
        """
        from .constants import (
            ACTION_ROLL, 
            BET_ACTION_TO_RANK,
            ACTION_PLACE_OASIS, 
            ACTION_PLACE_MIRAGE,
            GAME_WINNER_ACTION_TO_RANK, 
            GAME_LOSER_ACTION_TO_RANK
        )
        
        valid = []
        
        # 1. Roll: valid if dice remain
        if self.dice_remaining:
            valid.append(ACTION_ROLL)
        
        # 2. Leg Betting: valid if tiles remain for the camel at that rank
        rankings = self.get_rankings()
        for action, rank_idx in BET_ACTION_TO_RANK.items():
            camel = rankings[rank_idx]
            if self.tiles_remaining[camel]:
                valid.append(action)
        
        # 3. Desert Tiles: valid if the player can place one
        if hasattr(self, 'can_place_desert_tile'):
            if self.can_place_desert_tile(player):
                valid.append(ACTION_PLACE_OASIS)
                valid.append(ACTION_PLACE_MIRAGE)
                
        # 4. Game Winner Bets: valid if player hasn't bet on that camel yet
        if hasattr(self, 'can_bet_game_winner'):
            can_win = self.can_bet_game_winner(player)
            for action, rank_idx in GAME_WINNER_ACTION_TO_RANK.items():
                if rankings[rank_idx] in can_win:
                    valid.append(action)
                    
        # 5. Game Loser Bets: valid if player hasn't bet on that camel yet
        if hasattr(self, 'can_bet_game_loser'):
            can_lose = self.can_bet_game_loser(player)
            for action, rank_idx in GAME_LOSER_ACTION_TO_RANK.items():
                if rankings[rank_idx] in can_lose:
                    valid.append(action)
        
        return valid
    
    def copy(self) -> GameState:
        """
        Create a deep copy of this game state.
        
        Returns:
            A new GameState with all data deep-copied.
        """
        return GameState(
            board=[list(stack) for stack in self.board],
            dice_remaining=set(self.dice_remaining),
            tiles_remaining={c: list(tiles) for c, tiles in self.tiles_remaining.items()},
            player_bets=[list(bets) for bets in self.player_bets],
            player_coins=list(self.player_coins),
            desert_tiles=dict(self.desert_tiles),
            game_winner_bets=list(self.game_winner_bets),
            game_loser_bets=list(self.game_loser_bets),
            current_leg=self.current_leg,
            game_complete=self.game_complete,
            num_players=self.num_players,
        )
    
    def __repr__(self) -> str:
        """Pretty print the game state."""
        lines = ["GameState:"]
        
        # Board
        lines.append("  Board:")
        for space_idx, stack in enumerate(self.board):
            if stack:
                camel_str = ", ".join(CAMEL_NAMES[c] for c in stack)
                lines.append(f"    Space {space_idx + 1}: [{camel_str}] (bottom→top)")
        
        # Dice remaining
        dice_names = [CAMEL_NAMES[c] for c in sorted(self.dice_remaining)]
        lines.append(f"  Dice remaining: {dice_names}")
        
        # Tiles
        lines.append("  Tiles available:")
        for camel in CAMELS:
            tiles = self.tiles_remaining[camel]
            lines.append(f"    {CAMEL_NAMES[camel]}: {tiles}")
        
        # Player info
        lines.append("  Players:")
        for p in range(self.num_players):
            bets_str = ", ".join(
                f"{CAMEL_NAMES[c]}@{v}" for c, v in self.player_bets[p]
            ) or "none"
            lines.append(f"    P{p}: {self.player_coins[p]} coins, bets: {bets_str}")
        
        # Full game info
        if self.current_leg > 1 or self.desert_tiles or any(self.game_winner_bets) or any(self.game_loser_bets):
            lines.append(f"  Leg: {self.current_leg}")
            if self.desert_tiles:
                desert_str = ", ".join(
                    f"Space {s+1}: {'Oasis' if e == 1 else 'Mirage'} (P{o})"
                    for s, (o, e) in self.desert_tiles.items()
                )
                lines.append(f"  Desert tiles: {desert_str}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # Full game methods
    # =========================================================================
    
    def place_desert_tile(self, player: int, space: int, is_oasis: bool) -> None:
        """
        Place a desert tile on the board.
        
        Each player can only have one desert tile on the board at a time.
        Desert tiles cannot be placed on spaces 0, 1 (starting area) or on
        spaces with camels or other desert tiles.
        
        Args:
            player: Player index.
            space: Board space index (0-15).
            is_oasis: True for oasis (+1), False for mirage (-1).
            
        Raises:
            ValueError: If placement is invalid.
        """
        # Remove any existing tile from this player
        self.desert_tiles = {
            s: (o, e) for s, (o, e) in self.desert_tiles.items() if o != player
        }
        
        # Validate space
        if space in (0, 1):
            raise ValueError("Cannot place desert tile on starting spaces (1-2)")
        if self.board[space]:
            raise ValueError(f"Cannot place desert tile on space {space+1} - camels present")
        if space in self.desert_tiles:
            raise ValueError(f"Cannot place desert tile on space {space+1} - tile already present")
        
        # Place tile
        effect = OASIS_EFFECT if is_oasis else MIRAGE_EFFECT
        self.desert_tiles[space] = (player, effect)
    
    def can_place_desert_tile(self, player: int) -> list[int]:
        """
        Get list of valid spaces where player can place a desert tile.
        
        Returns:
            List of valid space indices.
        """
        # Remove current tile from consideration
        existing_tiles = {s for s, (o, e) in self.desert_tiles.items() if o != player}
        
        valid = []
        for space in range(2, NUM_SPACES):  # Skip spaces 0, 1
            if not self.board[space] and space not in existing_tiles:
                # Also check adjacent spaces don't have other tiles
                adjacent_has_tile = (
                    (space - 1) in existing_tiles or 
                    (space + 1) in existing_tiles
                )
                if not adjacent_has_tile:
                    valid.append(space)
        return valid
    
    def place_game_winner_bet(self, player: int, camel: int) -> None:
        """
        Place a secret bet on the overall game winner.
        
        Each player can only bet on each camel once for game winner.
        
        Args:
            player: Player index.
            camel: Camel index to bet on.
            
        Raises:
            ValueError: If already bet on this camel.
        """
        # Check if player already bet on this camel
        for p, c in self.game_winner_bets:
            if p == player and c == camel:
                raise ValueError(f"Player {player} already bet on {CAMEL_NAMES[camel]} for game winner")
        self.game_winner_bets.append((player, camel))
    
    def place_game_loser_bet(self, player: int, camel: int) -> None:
        """
        Place a secret bet on the overall game loser.
        
        Each player can only bet on each camel once for game loser.
        
        Args:
            player: Player index.
            camel: Camel index to bet on.
            
        Raises:
            ValueError: If already bet on this camel.
        """
        # Check if player already bet on this camel
        for p, c in self.game_loser_bets:
            if p == player and c == camel:
                raise ValueError(f"Player {player} already bet on {CAMEL_NAMES[camel]} for game loser")
        self.game_loser_bets.append((player, camel))
    
    def can_bet_game_winner(self, player: int) -> list[int]:
        """Get list of camels player can still bet on for game winner."""
        bet_camels = {c for p, c in self.game_winner_bets if p == player}
        return [c for c in CAMELS if c not in bet_camels]
    
    def can_bet_game_loser(self, player: int) -> list[int]:
        """Get list of camels player can still bet on for game loser."""
        bet_camels = {c for p, c in self.game_loser_bets if p == player}
        return [c for c in CAMELS if c not in bet_camels]
    
    def start_new_leg(self) -> None:
        """
        Reset leg state for a new leg.
        
        - Resets dice to pyramid
        - Resets leg betting tiles
        - Clears leg bets
        - Removes desert tiles
        - Increments leg counter
        """
        self.dice_remaining = set(CAMELS)
        self.tiles_remaining = {c: list(BET_TILE_VALUES) for c in CAMELS}
        self.player_bets = [[] for _ in range(self.num_players)]
        self.desert_tiles = {}
        self.current_leg += 1
    
    def get_leader_position(self) -> int:
        """Get the board position of the leading camel."""
        rankings = self.get_rankings()
        if rankings:
            pos, _ = self.get_camel_position(rankings[0])
            return pos
        return 0
    
    def is_game_complete(self) -> bool:
        """Check if any camel has crossed the finish line."""
        for camel in CAMELS:
            pos, _ = self.get_camel_position(camel)
            if pos >= FINISH_LINE - 1:  # Space 16 (0-indexed as 15)
                self.game_complete = True
                return True
        return False
    
    def score_game_end_bets(self) -> dict[int, int]:
        """
        Score all game-end bets at the end of the game.
        
        Bets are evaluated in order they were placed (FIFO).
        Correct bets get payouts: 8, 5, 3, 2, 1, 1, 1, 1...
        Wrong bets get GAME_END_BET_PENALTY (-1).
        
        Returns:
            Dict mapping player index to coins earned from game-end bets.
        """
        rankings = self.get_rankings()
        winner = rankings[0]
        loser = rankings[-1]
        
        deltas: dict[int, int] = {p: 0 for p in range(self.num_players)}
        
        # Collect all winner bets in order placed
        winner_bet_queue = self.game_winner_bets
        
        # Score winner bets
        correct_idx = 0
        for player, camel in winner_bet_queue:
            if camel == winner:
                payout = GAME_END_BET_PAYOUTS[min(correct_idx, len(GAME_END_BET_PAYOUTS) - 1)]
                deltas[player] += payout
                correct_idx += 1
            else:
                deltas[player] += GAME_END_BET_PENALTY
        
        # Collect all loser bets in order placed
        loser_bet_queue = self.game_loser_bets
        
        # Score loser bets
        correct_idx = 0
        for player, camel in loser_bet_queue:
            if camel == loser:
                payout = GAME_END_BET_PAYOUTS[min(correct_idx, len(GAME_END_BET_PAYOUTS) - 1)]
                deltas[player] += payout
                correct_idx += 1
            else:
                deltas[player] += GAME_END_BET_PENALTY
        
        # Add deltas to player coins
        for player, delta in deltas.items():
            self.player_coins[player] += delta
        
        return deltas

