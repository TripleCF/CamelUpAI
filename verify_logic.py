import gymnasium as gym
import numpy as np
from camel_leg.game_state import GameState
from camel_leg.constants import BET_ACTION_TO_RANK, ACTION_ROLL, CAMEL_NAMES

def verify_constants():
    print("Verifying constants...")
    try:
        from camel_leg.constants import BET_ACTION_TO_CAMEL
        print("❌ BET_ACTION_TO_CAMEL still exists!")
        return False
    except ImportError:
        print("✅ BET_ACTION_TO_CAMEL successfully removed.")
    
    if 1 not in BET_ACTION_TO_RANK:
        print("❌ BET_ACTION_TO_RANK missing key 1.")
        return False
    print("✅ Constants verified.")
    return True

def verify_mc_opponent():
    print("\nVerifying MC Opponent...")
    from camel_leg.monte_carlo import MCOpponent
    state = GameState.create_random_start()
    mc = MCOpponent(simulations=10)
    
    try:
        action = mc.get_action(state, 1)
        print(f"✅ MCOpponent generated action: {action}")
        
        if 1 <= action <= 5:
            # Verify it's rank-based
            # If it were color-based (old), it would be mapping to camel ID directly
            # Here we just verify it runs without crashing and returning valid range
            pass
        elif action == ACTION_ROLL:
            pass
        else:
             print(f"❌ Unknown action {action}")
             return False
    except Exception as e:
        print(f"❌ MC Opponent failed: {e}")
        return False
    
    print("✅ MC Opponent verified.")
    return True

def verify_env():
    print("\nVerifying CamelLegEnv...")
    try:
        from camel_leg.env import CamelLegEnv
        env = CamelLegEnv(opponent_type="mc", opponent_simulations=5)
        obs, _ = env.reset()
        
        # Take a betting action (Action 1 = Bet on 1st place)
        # We need to find if it's valid first
        masks = env.action_masks()
        action = 1
        if not masks[action]:
            action = 0 # Roll if bet not valid (unlikely at start but possible if no tiles)
        
        obs, reward, terminated, _, _ = env.step(action)
        print(f"✅ CamelLegEnv step successful. Action: {action}, Reward: {reward}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ CamelLegEnv failed: {e}")
        return False
    
    return True

def verify_full_env():
    print("\nVerifying CamelFullGameEnv...")
    try:
        from camel_leg.full_game_env import CamelFullGameEnv
        env = CamelFullGameEnv(opponent_type="mc", opponent_simulations=5)
        obs, _ = env.reset()
        
        action = 0 # Roll
        obs, reward, terminated, _, _ = env.step(action)
        print(f"✅ CamelFullGameEnv step successful. Action: 0")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ CamelFullGameEnv failed: {e}")
        return False
    return True

if __name__ == "__main__":
    c = verify_constants()
    m = verify_mc_opponent()
    e = verify_env()
    f = verify_full_env()
    
    if c and m and e and f:
        print("\n✅ ALL CHECKS PASSED")
        exit(0)
    else:
        print("\n❌ SOME CHECKS FAILED")
        exit(1)
