"""
PPO training script for Camel Up RL agent.

Supports both single-leg and full game training.
"""

import argparse
from pathlib import Path

import numpy as np


def make_env(opponent_type: str, opponent_simulations: int, seed: int):
    """Create a wrapped Camel Up environment."""
    from camel_leg.env import CamelLegEnv
    
    def _init():
        env = CamelLegEnv(
            opponent_type=opponent_type,
            opponent_simulations=opponent_simulations,
            seed=seed,
        )
        return env
    
    return _init


def train(
    total_timesteps: int = 100_000,
    opponent_type: str = "mc",
    opponent_simulations: int = 100,
    save_path: str = "camel_ppo_agent",
    load_path: str = None,
    seed: int = None,
    log_dir: str = "./logs",
    render: bool = False,
):
    """
    Train a PPO agent for Camel Up.
    
    Args:
        total_timesteps: Total training timesteps.
        opponent_type: "mc" for Monte Carlo, "random" for random opponents.
        opponent_simulations: Number of MC simulations per opponent action.
        save_path: Path to save the trained model.
        load_path: Path to an existing model to continue training from.
        seed: Random seed.
        log_dir: TensorBoard log directory.
        render: Whether to render the environment during training.
    """
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("Error: sb3-contrib and stable-baselines3 are required for training.")
        print("Install with: pip install sb3-contrib stable-baselines3")
        return
    
    from camel_leg.env import CamelLegEnv
    
    print(f"Creating environment with {opponent_type} opponents...")
    
    # Create environment
    env = CamelLegEnv(
        opponent_type=opponent_type,
        opponent_simulations=opponent_simulations,
        seed=seed,
        render_mode="human" if render else None,
    )
    
    # Wrap with ActionMasker for MaskablePPO
    def mask_fn(env):
        return env.action_masks()
    
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)
    
    print(f"Training for {total_timesteps} timesteps...")
    
    if load_path and Path(load_path).exists():
        print(f"Loading existing model from {load_path}...")
        model = MaskablePPO.load(load_path, env=env)
        # Update tensorboard log if provided
        if log_dir:
            model.tensorboard_log = log_dir
    else:
        if load_path:
            print(f"Warning: Model path {load_path} not found. Starting from scratch.")
        
        print("Creating new model...")
        # Create model
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,  
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
    
    # Train without evaluation callback (standard EvalCallback has issues with MaskablePPO)
    # Use the separate 'eval' command after training to evaluate the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")
    
    # Save final model
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model


def evaluate(
    model_path: str,
    num_episodes: int = 100,
    opponent_type: str = "mc",
    opponent_simulations: int = 100,
    seed: int = None,
    render: bool = False,
):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the saved model.
        num_episodes: Number of episodes to evaluate.
        opponent_type: Type of opponents.
        opponent_simulations: MC simulations for opponents.
        seed: Random seed.
        render: Whether to render games.
    """
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError:
        print("Error: sb3-contrib is required for evaluation.")
        return
    
    from camel_leg.env import CamelLegEnv
    
    # Load model
    model = MaskablePPO.load(model_path)
    
    # Create environment
    render_mode = "human" if render else None
    env = CamelLegEnv(
        opponent_type=opponent_type,
        opponent_simulations=opponent_simulations,
        seed=seed,
        render_mode=render_mode,
    )
    
    def mask_fn(env):
        return env.action_masks()
    
    env = ActionMasker(env, mask_fn)
    
    rewards = []
    wins = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get action masks for MaskablePPO
            action_masks = env.unwrapped.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            # Convert numpy array to int (predict returns np.ndarray)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
        
        # Check if agent won (most coins)
        final_coins = env.unwrapped.game_state.player_coins
        if final_coins[0] == max(final_coins):
            wins += 1
    
    print(f"\nEvaluation Results ({num_episodes} episodes):")
    print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Win rate: {wins/num_episodes*100:.1f}%")
    print(f"  Min/Max reward: {min(rewards):.2f} / {max(rewards):.2f}")
    
    return rewards


def train_full_game(
    total_timesteps: int = 500_000,
    num_opponents: int = 3,
    opponent_type: str = "random",
    opponent_simulations: int = 50,
    reward_mode: str = "win",
    save_path: str = "camel_full_game_agent",
    load_path: str = None,
    seed: int = None,
    log_dir: str = "./logs",
    render: bool = False,
):
    """
    Train a PPO agent for full Camel Up game.
    
    Args:
        total_timesteps: Total training timesteps.
        num_opponents: Number of opponents (1-7).
        opponent_type: "mc", "random", or "mixed".
        opponent_simulations: MC simulations per opponent action.
        reward_mode: "win", "coins", or "combined".
        save_path: Path to save the trained model.
        load_path: Path to existing model to continue training.
        seed: Random seed.
        log_dir: TensorBoard log directory.
        render: Whether to render during training.
    """
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("Error: sb3-contrib and stable-baselines3 required.")
        return
    
    from camel_leg.full_game_env import CamelFullGameEnv
    
    print(f"Creating full game environment...")
    print(f"  Opponents: {num_opponents} ({opponent_type})")
    print(f"  Reward mode: {reward_mode}")
    
    env = CamelFullGameEnv(
        num_opponents=num_opponents,
        opponent_type=opponent_type,
        opponent_simulations=opponent_simulations,
        reward_mode=reward_mode,
        seed=seed,
        render_mode="human" if render else None,
    )
    
    def mask_fn(env):
        return env.action_masks()
    
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)
    
    print(f"Training for {total_timesteps} timesteps...")
    
    if load_path and Path(load_path).exists():
        print(f"Loading existing model from {load_path}...")
        model = MaskablePPO.load(load_path, env=env)
        if log_dir:
            model.tensorboard_log = log_dir
    else:
        if load_path:
            print(f"Warning: Model path {load_path} not found. Starting fresh.")
        
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )
    
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving...")
    
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model


def evaluate_full_game(
    model_path: str,
    num_episodes: int = 100,
    num_opponents: int = 3,
    opponent_type: str = "random",
    opponent_simulations: int = 50,
    seed: int = None,
    render: bool = False,
):
    """Evaluate a trained full game model."""
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
    except ImportError:
        print("Error: sb3-contrib required.")
        return
    
    from camel_leg.full_game_env import CamelFullGameEnv
    
    model = MaskablePPO.load(model_path)
    
    env = CamelFullGameEnv(
        num_opponents=num_opponents,
        opponent_type=opponent_type,
        opponent_simulations=opponent_simulations,
        reward_mode="win",
        seed=seed,
        render_mode="human" if render else None,
    )
    
    def mask_fn(env):
        return env.action_masks()
    
    env = ActionMasker(env, mask_fn)
    
    wins = 0
    total_coins = 0
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action_masks = env.unwrapped.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        if reward > 0:
            wins += 1
        total_coins += info.get('player_coins', [0])[env.unwrapped.agent_index]
    
    print(f"\nFull Game Evaluation ({num_episodes} games):")
    print(f"  Win rate: {wins/num_episodes*100:.1f}%")
    print(f"  Avg coins: {total_coins/num_episodes:.1f}")
    
    return wins, total_coins


def analyze_logs(log_dir: str = "./logs"):
    """
    Analyze TensorBoard logs to see reward trends.
    """
    from tensorboard.backend.event_processing import event_accumulator
    import os
    
    # Find all MaskablePPO log directories
    log_path = Path(log_dir)
    dirs = [d for d in log_path.iterdir() if d.is_dir() and d.name.startswith("MaskablePPO_")]
    
    if not dirs:
        print(f"No logs found in {log_dir}")
        return
    
    # Sort by modification time to get the latest
    latest_dir = max(dirs, key=lambda d: d.stat().st_mtime)
    print(f"Analyzing latest log: {latest_dir}")
    
    ea = event_accumulator.EventAccumulator(str(latest_dir))
    ea.Reload()
    
    if 'rollout/ep_rew_mean' not in ea.Tags()['scalars']:
        print("Could not find 'rollout/ep_rew_mean' in logs.")
        # Try to find any rewards
        available = [t for t in ea.Tags()['scalars'] if 'rew' in t]
        if available:
            print(f"Available reward tags: {available}")
        return

    rewards = ea.Scalars('rollout/ep_rew_mean')
    if not rewards:
        print("Log is empty.")
        return

    steps = [r.step for r in rewards]
    values = [r.value for r in rewards]
    
    start_val = np.mean(values[:max(1, len(values)//10)])
    end_val = np.mean(values[-max(1, len(values)//10):])
    
    improvement = end_val - start_val
    percent = (improvement / abs(start_val) * 100) if start_val != 0 else 0
    
    print("\n--- Training Progress Analysis ---")
    print(f"Total data points: {len(values)}")
    print(f"Starting Mean Reward: {start_val:.2f}")
    print(f"Current Mean Reward:  {end_val:.2f}")
    print(f"Net Improvement:      {improvement:+.2f} ({percent:+.1f}%)")
    
    if len(values) > 1:
        # Calculate slope (linear regression)
        z = np.polyfit(steps, values, 1)
        slope = z[0]
        print(f"Trend (Slope):        {slope*1000:+.6f} per 1k steps")
        
        if slope > 0:
            print("Status: LEARNING (Positive trend)")
        elif slope < -0.0001:
            print("Status: REGRESSING (Negative trend)")
        else:
            print("Status: STABLE / NOISY")
    
    return values


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(description="Train/evaluate Camel Up RL agent")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Total training timesteps"
    )
    train_parser.add_argument(
        "--opponent", type=str, default="mc", choices=["mc", "random"],
        help="Opponent type"
    )
    train_parser.add_argument(
        "--simulations", type=int, default=100,
        help="MC simulations per opponent action"
    )
    train_parser.add_argument(
        "--save", type=str, default="camel_ppo_agent",
        help="Save path for model"
    )
    train_parser.add_argument(
        "--load", type=str, default=None,
        help="Path to an existing model to continue training from"
    )
    train_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    train_parser.add_argument(
        "--render", action="store_true",
        help="Render games during training"
    )
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument(
        "model", type=str,
        help="Path to saved model"
    )
    eval_parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of evaluation episodes"
    )
    eval_parser.add_argument(
        "--opponent", type=str, default="mc", choices=["mc", "random"],
        help="Opponent type"
    )
    eval_parser.add_argument(
        "--simulations", type=int, default=100,
        help="MC simulations per opponent action"
    )
    eval_parser.add_argument(
        "--render", action="store_true",
        help="Render games"
    )
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Launch interactive scenario builder")
    interactive_parser.add_argument(
        "--model", type=str, default=None,
        help="Path to saved model (optional, for model predictions)"
    )
    interactive_parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to run server on"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze training logs")
    analyze_parser.add_argument(
        "--log_dir", type=str, default="./logs",
        help="Directory containing TensorBoard logs"
    )
    
    # Train full game command
    train_full_parser = subparsers.add_parser("train-full", help="Train agent for full game")
    train_full_parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps"
    )
    train_full_parser.add_argument(
        "--opponents", type=int, default=3,
        help="Number of opponents (1-7)"
    )
    train_full_parser.add_argument(
        "--opponent", type=str, default="random", choices=["mc", "random", "mixed"],
        help="Opponent type"
    )
    train_full_parser.add_argument(
        "--simulations", type=int, default=50,
        help="MC simulations per opponent"
    )
    train_full_parser.add_argument(
        "--reward", type=str, default="win", choices=["win", "coins", "combined"],
        help="Reward mode"
    )
    train_full_parser.add_argument(
        "--save", type=str, default="camel_full_game_agent",
        help="Save path"
    )
    train_full_parser.add_argument(
        "--load", type=str, default=None,
        help="Load existing model"
    )
    train_full_parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    train_full_parser.add_argument(
        "--render", action="store_true",
        help="Render during training"
    )
    
    # Eval full game command
    eval_full_parser = subparsers.add_parser("eval-full", help="Evaluate full game agent")
    eval_full_parser.add_argument(
        "model", type=str,
        help="Path to saved model"
    )
    eval_full_parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of games"
    )
    eval_full_parser.add_argument(
        "--opponents", type=int, default=3,
        help="Number of opponents"
    )
    eval_full_parser.add_argument(
        "--opponent", type=str, default="mc", choices=["mc", "random", "mixed"],
        help="Opponent type"
    )
    eval_full_parser.add_argument(
        "--simulations", type=int, default=50,
        help="MC simulations"
    )
    eval_full_parser.add_argument(
        "--render", action="store_true",
        help="Render games"
    )
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(
            total_timesteps=args.timesteps,
            opponent_type=args.opponent,
            opponent_simulations=args.simulations,
            save_path=args.save,
            load_path=args.load,
            seed=args.seed,
            render=args.render,
        )
    elif args.command == "eval":
        evaluate(
            model_path=args.model,
            num_episodes=args.episodes,
            opponent_type=args.opponent,
            opponent_simulations=args.simulations,
            render=args.render,
        )
    elif args.command == "interactive":
        from camel_leg.interactive_server import run_server
        run_server(model_path=args.model, port=args.port)
    elif args.command == "analyze":
        analyze_logs(log_dir=args.log_dir)
    elif args.command == "train-full":
        train_full_game(
            total_timesteps=args.timesteps,
            num_opponents=args.opponents,
            opponent_type=args.opponent,
            opponent_simulations=args.simulations,
            reward_mode=args.reward,
            save_path=args.save,
            load_path=args.load,
            seed=args.seed,
            render=args.render,
        )
    elif args.command == "eval-full":
        evaluate_full_game(
            model_path=args.model,
            num_episodes=args.episodes,
            num_opponents=args.opponents,
            opponent_type=args.opponent,
            opponent_simulations=args.simulations,
            render=args.render,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
