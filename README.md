# Mancala Reinforcement Learning

Designing, training, and evaluating a reinforcement learning agent for the game of **Mancala** using Q-learning with hybrid exploration strategies and dynamic parameter decay.

## Overview

This project implements a Mancala game environment with a reinforcement learning agent trained using Q-learning. The agent employs an advanced action selection strategy combining softmax and epsilon-greedy methods, along with dynamic parameter decay for epsilon, alpha (learning rate), and temperature.

**Key Highlights:**
- Advanced Q-learning with hybrid action selection
- Comprehensive training analysis across multiple runs
- Interactive gameplay against trained AI
- Performance visualization and consistency metrics
- Model persistence for iterative training

## Features

### Game Modes
- **Human vs Human**: Classic two-player Mancala
- **Human vs AI**: Play against the trained reinforcement learning agent
- **AI Training Mode**: Self-play for agent training

### Advanced RL Implementation
- **Hybrid Action Selection**: Combines softmax and epsilon-greedy strategies
- **Parameter Decay**: Dynamic decay for epsilon, alpha, and temperature
- **Sophisticated Reward System**:
  - +2 points per marble gained in player's mancala
  - +10 points per stone captured from opponent
  - +1000 points for winning the game
  - -1000 points for losing the game
- **State Space Representation**: Efficient board state encoding
- **Model Persistence**: Save and load trained agents via pickle

### Training Features
- Multiple training runs for consistency analysis
- Checkpoint saving every 25,000 games
- Moving average and win rate visualization
- Cross-run consistency plots
- Configurable hyperparameters

## Installation

### Prerequisites
- Python 3.6+
- NumPy
- Matplotlib

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd "mancala reinforcement learning (python)"
```

2. Install dependencies:
```bash
pip install numpy matplotlib
```

## Quick Start

### Train a New Agent

```bash
cd mancala
python train_mancala_agent.py
```

This will:
- Train 5 independent agents for 1,000,000 games each
- Save checkpoints every 25,000 games
- Generate performance plots and consistency analysis
- Save trained models to the `model/` directory

### Play Against the Agent

```bash
cd mancala
python play_mancala.py
```

Follow the prompts to:
- Choose whether Player 1 is human or computer
- Choose whether Player 2 is human or computer
- Play the game by selecting pocket numbers

## Project Structure

```
mancala-reinforcement-learning/
├── mancala/
│   ├── mancala.py              # Core game engine and board logic
│   ├── agent.py                # Q-learning agent implementation
│   ├── train_mancala_agent.py  # Training script with analysis
│   ├── play_mancala.py         # Interactive gameplay interface
│   └── tests/                  # Test files
├── model/                      # Saved agent models (.pkl files)
├── figures (softmax + rewards + parameter decays)/
│   ├── Run1MovingAvg&WinRate.png
│   ├── Run2MovingAvg&WinRate.png
│   ├── ...
│   └── consistency_plot.png
├── ECS 170 Project Presentation.pdf
├── README.md
├── LICENSE
└── CODE_OF_CONDUCT.md
```

## Algorithm Details

### Q-Learning Implementation

The agent uses **temporal difference Q-learning** with the following update rule:

```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

Where:
- `α` (alpha): Learning rate (starts at 0.4, decays to 0.01)
- `γ` (gamma): Discount factor (0.5)
- `r`: Reward received
- `s`: Current state
- `a`: Action taken
- `s'`: Next state

### Hybrid Action Selection

The agent uses a combination of **epsilon-greedy** and **softmax** strategies:

1. **Epsilon-Greedy Component**: With probability ε, choose a random action (exploration)
2. **Softmax Component**: Otherwise, select actions probabilistically based on Q-values:

```
P(a) = exp(Q(a)/T) / Σ exp(Q(a')/T)
```

Where `T` is the temperature parameter controlling exploration vs exploitation.

### Parameter Decay

All three key parameters decay over training:
- **Epsilon**: 1.0 → 0.05 (decay rate: 0.999)
- **Alpha**: 0.4 → 0.01 (decay rate: 0.999)
- **Temperature**: 1.0 → 0.1 (decay rate: 0.999)

This allows for high exploration early in training and exploitation of learned strategies later.

## Training

### Default Training Configuration

```python
train_agent(
    n_games=1000000,              # Total games per run
    games_per_checkpoint=25000,   # Save frequency
    initial_epsilon=1.0,
    min_epsilon=0.05,
    epsilon_decay=0.999,
    initial_alpha=0.4,
    min_alpha=0.01,
    alpha_decay=0.999,
    initial_temperature=1.0,
    min_temperature=0.1,
    temperature_decay=0.999
)
```

### Training Output

The training script generates:
- **5 trained agent models** saved as `.pkl` files
- **Moving average plots** for each run showing outcome trends
- **Win rate plots** tracking performance over time
- **Consistency plot** comparing all 5 runs

### Customizing Training

Edit `train_mancala_agent.py` to modify:
- Number of games
- Checkpoint frequency
- Decay rates and initial values
- Reward structure

## 🎲 Playing Against the Agent

When running `play_mancala.py`:

1. The game loads the trained agent from `model/mancala_agent.pkl`
2. You'll be prompted to choose human or computer for each player
3. The board is displayed with pocket numbers
4. Enter the pocket number (0-5 for Player 1, 7-12 for Player 2) to make your move
5. The game follows standard Mancala rules with captures and extra turns

### Mancala Rules
- Players take turns picking up all stones from one of their pockets
- Stones are distributed counter-clockwise, one per pocket
- If the last stone lands in your mancala, you get another turn
- If the last stone lands in an empty pocket on your side, you capture that stone and all stones in the opposite pocket
- The game ends when one side has no stones left
- The player with the most stones in their mancala wins

## Results

The trained agent demonstrates:
- Consistent learning across multiple independent runs
- Improved win rates over training duration
- Strategic play including captures and extra turn optimization
- Competitive gameplay against human players

See the `figures (softmax + rewards + parameter decays)/` directory for detailed performance visualizations.

## 👥 Contributors

This project was developed by a team of 8 students for ECS 170:

- **Angelina Cottone**
- **Nick Gomez**
- **Harris Habib**
- **Mythri Kulkarni**
- **Aatish Lobo**
- **Andrew Ortega**
- **Shriya Rudrashetty**
- **Alyssa Ann Toledo**

### Branch Structure

Each branch contains a specific team member's experimental adjustments to the baseline code. The `master` branch contains the final model with the complete reward system, all figures/plots, and presentation materials. See the Contributions section (Section 7) of the project report for detailed branch-to-contributor mapping.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Code of Conduct

Please review our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

---

**Course**: ECS 170 - Artificial Intelligence
**Project**: Mancala Reinforcement Learning Implementation
