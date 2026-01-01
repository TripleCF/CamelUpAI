
import sys
import os
import numpy as np

# Ensure we can import camel_leg
sys.path.append(os.getcwd())

from camel_leg.full_game_env import CamelFullGameEnv

def verify_env():
    print("=== Verifying Environment ===")
    try:
        env = CamelFullGameEnv()
        env.reset()
        print("✅ Environment created and reset successfully.")
        
        obs_space = env.observation_space
        print(f"Observation Space keys: {obs_space.keys()}")
        
        # Check specific keys
        if "game_winner_bets_placed" in obs_space.keys():
            shape = obs_space["game_winner_bets_placed"].shape
            print(f"✅ game_winner_bets_placed shape: {shape}")
            assert shape == (1,), f"Expected (1,), got {shape}"
        else:
            print("❌ MISSING game_winner_bets_placed")
            exit(1)
            
        print("✅ Verification Complete")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    verify_env()
