import time
import numpy as np
from camel_leg.game_state import GameState
from camel_leg.monte_carlo import monte_carlo_action, MCOpponent

def benchmark():
    state = GameState.create_random_start()
    # No rolls yet, so all dice remaining
    
    print("\n--- Benchmarking Optimized MC Action Selection ---")
    print(f"Dice remaining: {len(state.dice_remaining)}")
    print(f"Leader position: {state.get_leader_position()}")
    
    sims = 1000
    iterations = 20
    
    start_time = time.perf_counter()
    for i in range(iterations):
        best, values = monte_carlo_action(state, player=0, simulations=sims, return_values=True)
        if i == 0:
            print(f"Top EV actions: {sorted(values.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    
    print(f"\nTotal time for {iterations} decisions ({sims} sims/batch): {total_time:.4f}s")
    print(f"Average time per decision: {avg_time*1000:.2f}ms")

if __name__ == "__main__":
    benchmark()
