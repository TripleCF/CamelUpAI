import time
from camel_leg.game_state import GameState
from camel_leg.monte_carlo import MCOpponent
from camel_leg.constants import CAMEL_NAMES
from camel_leg.interactive_server import get_mc_recommendation
from itertools import permutations, product
from camel_leg.constants import CAMELS
from camel_leg.game_state import GameState
import numpy as np


def main():
    # Create a random game state
    state = GameState.create_random_start()
    state.board[0] = [0, 2]
    state.board[1] = [1]    
    state.board[2] = [3, 4]
    
    print("\n--- Current Game State ---")
    print(f"Dice remaining: {state.dice_remaining}")
    for i, stack in enumerate(state.board):
        if stack:
            names = [CAMEL_NAMES[c] for c in stack]
            print(f"Space {i+1}: {names}")
    
    iterations = 100
    print(f"\nCalculating exact win probabilities {iterations} times...")
    
    start_time = time.perf_counter()
    for _ in range(iterations):
        probs = simulate_probabilities(state)
        #probs = exact_win_probabilities(state)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    
    print("\nResults (Win %, 2nd Place %):")
    for camel_idx, (p1, p2) in probs.items():
        print(f"{CAMEL_NAMES[camel_idx]:<7}: {p1:>6.1%} / {p2:>6.1%}")
    
    print(f"\n--- Benchmark Results ---")
    print(f"Total time for {iterations} runs: {total_time:.4f} seconds")
    print(f"Average time per run: {avg_time*1000:.2f} ms")
    print()
    

def exact_win_probabilities(state: GameState) -> dict[int, tuple[float, float]]:
    """Berechnet exakte P(1st) und P(2nd) für jedes Kamel."""
    dice_remaining = list(state.dice_remaining)
    if not dice_remaining:
        # Leg bereits vorbei
        rankings = state.get_rankings()
        return {c: (1.0, 0.0) if c == rankings[0] else (0.0, 1.0) if c == rankings[1] else (0.0, 0.0) for c in CAMELS}
    
    n = len(dice_remaining)
    rolls = list(product([1, 2, 3], repeat=n))  # Alle Augenzahl-Kombinationen
    orderings = list(permutations(dice_remaining))  # Alle Würfelreihenfolgen
    
    win_counts = {c: 0 for c in CAMELS}
    second_counts = {c: 0 for c in CAMELS}
    total = len(rolls) * len(orderings)
    
    for ordering in orderings:
        for roll_values in rolls:
            sim_state = state.copy()
            for camel, distance in zip(ordering, roll_values):
                sim_state.move_camel(camel, distance)
            rankings = sim_state.get_rankings()
            win_counts[rankings[0]] += 1
            second_counts[rankings[1]] += 1
    
    return {c: (win_counts[c] / total, second_counts[c] / total) for c in CAMELS}

def simulate_probabilities(state: GameState, simulations: int = 100000) -> dict[int, tuple[float, float]]:
    """Get Monte Carlo's recommended action with both immediate and strategic EVs."""
    try:
        opponent = MCOpponent(simulations=simulations)
        
        valid_actions = state.get_valid_actions(0)
        if not valid_actions:
            return {"error": "No valid actions"}
        
        # Run simulations to get win probabilities for each camel
        win_probs = {c: 0.0 for c in CAMELS}
        second_probs = {c: 0.0 for c in CAMELS}
        rng = np.random.default_rng()
        
        for _ in range(simulations):
            sim_state = state.copy()
            while not sim_state.is_leg_complete():
                sim_state.roll_die(rng)
            rankings = sim_state.get_rankings()
            win_probs[rankings[0]] += 1
            second_probs[rankings[1]] += 1
        
        for c in CAMELS:
            win_probs[c] /= simulations
            second_probs[c] /= simulations

        return {c: (win_probs[c], second_probs[c]) for c in CAMELS}

    except Exception as e:
        print(f"Error in get_mc_recommendation: {e}")
        return {c: (0.0, 0.0) for c in CAMELS}

if __name__ == "__main__":
    main()

