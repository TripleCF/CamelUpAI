# Camel Up RL: Training and Evaluation Walkthrough

This guide explains how to train, evaluate, and interact with the Reinforcement Learning agent for the Camel Up game (leg segment).

## 1. Setup

Before training, ensure you have the necessary dependencies installed. The training process requires `stable-baselines3` and `sb3-contrib`.

```bash
pip install -r camel_leg/requirements.txt
# Additionally install RL libraries if not already installed
pip install stable-baselines3 sb3-contrib tensorboard
```

## 2. Training the Agent

To train a new PPO agent, use the `train` command.

### Basic Training
This command trains an agent for 100,000 steps against an optimal opponent (Monte Carlo simulations).

```bash
python3 -m camel_leg.train train --timesteps 100000 --save camel_ppo_agent
```

### Fast Training (Random Opponent)
If you want to quickly test the training pipeline without waiting for Monte Carlo simulations:

```bash
python3 -m camel_leg.train train --timesteps 50000 --opponent random --save quick_agent
```

### Customizing Training
- `--timesteps`: Total steps to train (e.g., `500000` for better performance).
- `--simulations`: Number of MC simulations the opponent runs per move (default: 100). Higher is better but slower.
- `--seed`: Set a random seed for reproducibility.

## 3. Evaluating the Agent

After training, you can evaluate your agent's performance (win rate and rewards).

```bash
python3 -m camel_leg.train eval camel_ppo_agent.zip --episodes 100
```

### Visual Evaluation
To see the agent play in the console (if render mode is supported):

```bash
python3 -m camel_leg.train eval camel_ppo_agent.zip --episodes 5 --render
```

## 4. Interactive Scenario Builder

You can launch a web-based interactive tool to visualize the game state and see the agent's predictions in real-time.

```bash
python3 -m camel_leg.train interactive --model camel_ppo_agent.zip --port 5000
```

Then open your browser at `http://localhost:5000`.

## 5. Directory Structure
- `camel_leg/env.py`: The Gymnasium environment logic.
- `camel_leg/game_state.py`: Core game rules and state management.
- `camel_leg/train.py`: CLI entry point for all operations.
- `logs/`: TensorBoard logs for monitoring training progress.
- `best_model/`: Storage for the best performing weights.

## 6. Monitoring Progress
You can monitor training in real-time using TensorBoard:

```bash
tensorboard --logdir ./logs
```
