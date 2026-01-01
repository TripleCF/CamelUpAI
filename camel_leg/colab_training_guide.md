# 🐪 Camel Up RL: Google Colab Training Guide

This guide explains how to run the Camel Up reinforcement learning training on Google Colab with free GPU/TPU acceleration.

## 📋 Prerequisites

Before starting on Colab, you need to make your code accessible:

### Option A: GitHub Repository (Recommended)
1. Push your `Simulator` project to a GitHub repository
2. You can use either public or private repo

### Option B: Upload as ZIP
1. Compress the `camel_leg` folder into a ZIP file
2. Upload directly to Colab when ready

---

## 🚀 Colab Setup Notebook

Create a new Colab notebook and run the following cells:

### Cell 1: Enable GPU Runtime
First, go to **Runtime → Change runtime type → GPU (T4 or higher recommended)**

Then verify GPU is available:
```python
!nvidia-smi
```

### Cell 2: Clone Your Repository
```python
# Option A: Clone from GitHub
!git clone https://github.com/YOUR_USERNAME/Simulator.git
%cd Simulator

# Option B: Upload ZIP (uncomment if using this method)
# from google.colab import files
# uploaded = files.upload()  # Upload your camel_leg.zip
# !unzip camel_leg.zip
```

### Cell 3: Install Dependencies
```python
# Install core dependencies
!pip install numpy gymnasium flask

# Install RL libraries (required for training)
!pip install stable-baselines3 sb3-contrib tensorboard
```

### Cell 4: Verify Installation
```python
import gymnasium
import stable_baselines3
import sb3_contrib
print(f"Gymnasium: {gymnasium.__version__}")
print(f"SB3: {stable_baselines3.__version__}")
print(f"SB3-Contrib: {sb3_contrib.__version__}")
```

---

## 🎯 Training Commands

### Full Game Training (Recommended)

This trains the agent to play the complete Camel Up game:

```python
# Train against random opponents (faster, good for initial training)
!python3 -m camel_leg.train train-full \
    --timesteps 500000 \
    --opponents 3 \
    --opponent random \
    --reward win \
    --save camel_full_game_random_500k
```

```python
# Train against mixed opponents (random + MC, more robust)
!python3 -m camel_leg.train train-full \
    --timesteps 500000 \
    --opponents 3 \
    --opponent mixed \
    --reward win \
    --save camel_full_game_mixed_500k
```

```python
# Train against Monte Carlo opponents (slower but strongest training signal)
!python3 -m camel_leg.train train-full \
    --timesteps 300000 \
    --opponents 3 \
    --opponent mc \
    --simulations 50 \
    --reward win \
    --save camel_full_game_mc_300k
```

### Continue Training from Existing Model
If you have a model to continue training from:

```python
# Upload your existing model first
from google.colab import files
uploaded = files.upload()  # Upload your model.zip file

# Continue training from that checkpoint
!python3 -m camel_leg.train train-full \
    --timesteps 200000 \
    --opponents 3 \
    --opponent mixed \
    --load your_model.zip \
    --save your_model_continued
```

### Legacy: Single-Leg Training
For training the simpler single-leg only model:

```python
!python3 -m camel_leg.train train \
    --timesteps 100000 \
    --opponent mc \
    --save camel_leg_agent
```

---

## 📊 Monitoring Training

### TensorBoard in Colab
```python
# Load TensorBoard extension
%load_ext tensorboard

# Start TensorBoard (run in a separate cell while training)
%tensorboard --logdir ./logs
```

### Analyze Training Progress
```python
!python3 -m camel_leg.train analyze --log_dir ./logs
```

---

## 📥 Downloading Your Trained Model

After training completes, download the model to your local machine:

```python
from google.colab import files

# Download the trained model
files.download('camel_full_game_random_500k.zip')

# Also download logs if needed
!zip -r training_logs.zip logs/
files.download('training_logs.zip')
```

---

## ⚡ Training Time Estimates

| Configuration | GPU (T4) | CPU Only |
|---------------|----------|----------|
| 100k steps, random opp | ~5 min | ~15 min |
| 500k steps, random opp | ~25 min | ~1.5 hr |
| 500k steps, mixed opp | ~45 min | ~3 hr |
| 500k steps, MC opp (50 sims) | ~2-3 hr | ~10+ hr |

> **Tip**: Start with random opponents for initial training, then continue with mixed/MC opponents to refine the agent's strategy.

---

## 🔧 Recommended Training Pipeline

Here's a suggested multi-stage training approach:

### Stage 1: Initial Learning (Random)
```python
!python3 -m camel_leg.train train-full \
    --timesteps 500000 \
    --opponents 3 \
    --opponent random \
    --reward win \
    --save stage1_random
```

### Stage 2: Refinement (Mixed)
```python
!python3 -m camel_leg.train train-full \
    --timesteps 500000 \
    --opponents 3 \
    --opponent mixed \
    --load stage1_random.zip \
    --save stage2_mixed
```

### Stage 3: Polish (MC)
```python
!python3 -m camel_leg.train train-full \
    --timesteps 300000 \
    --opponents 3 \
    --opponent mc \
    --simulations 50 \
    --load stage2_mixed.zip \
    --save stage3_final
```

### Evaluate Final Model
```python
!python3 -m camel_leg.train eval-full stage3_final.zip \
    --episodes 200 \
    --opponents 3 \
    --opponent mc \
    --simulations 100
```

---

## ⚠️ Common Issues & Solutions

### 1. Runtime Disconnection
Colab free tier disconnects after ~90 min of inactivity. Solutions:
- Keep the browser tab active
- Save checkpoints frequently using `--save` 
- Use Colab Pro for longer runtimes

### 2. Out of Memory
If you get OOM errors:
```python
# Reduce batch size by modifying train.py or use smaller simulations
!python3 -m camel_leg.train train-full \
    --timesteps 500000 \
    --simulations 25 \  # Reduced from 50
    --save model_small_batch
```

### 3. Module Not Found
Ensure you're in the correct directory:
```python
import os
print(os.getcwd())  # Should show .../Simulator
%cd /content/Simulator  # Navigate if needed
```

---

## 📁 Complete Colab Notebook Template

Here's a complete notebook you can copy-paste:

```python
# === CELL 1: Setup ===
!git clone https://github.com/YOUR_USERNAME/Simulator.git
%cd Simulator
!pip install numpy gymnasium flask stable-baselines3 sb3-contrib tensorboard

# === CELL 2: Verify ===
import gymnasium, stable_baselines3, sb3_contrib
print(f"All packages installed successfully!")

# === CELL 3: Train ===
!python3 -m camel_leg.train train-full \
    --timesteps 500000 \
    --opponents 3 \
    --opponent random \
    --reward win \
    --save my_trained_model

# === CELL 4: Evaluate ===
!python3 -m camel_leg.train eval-full my_trained_model.zip --episodes 100

# === CELL 5: Download ===
from google.colab import files
files.download('my_trained_model.zip')
```

---

## 💡 Tips for Best Results

1. **GPU is crucial**: Training is ~5-10x faster on GPU vs CPU
2. **Start simple**: Begin with random opponents before moving to MC
3. **Save frequently**: Colab can disconnect unexpectedly
4. **Monitor loss**: Use TensorBoard to track training progress
5. **Experiment with reward modes**: Try `--reward combined` for different learning signals
