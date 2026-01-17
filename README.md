# Mancala Reinforcement Learning

An experimental reinforcement learning study of the game **Mancala**, focused on reward shaping, exploration strategies, and state-space reduction for improved policy learning.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

---

## TL;DR
- Built and iteratively improved a reinforcement learning agent for Mancala
- Increased win rate from ~30% (baseline) to **72.59%** through systematic experimentation
- Demonstrated that **state aggregation and reward design** dominate raw algorithmic complexity
- Evaluated epsilon decay, alpha decay, softmax exploration, Double Q-learning, and reward scaling
- Identified tradeoffs between convergence speed, stability, and policy consistency

---

## Project Overview

This project applies **reinforcement learning** to the game of Mancala to study how learning dynamics are affected by reward design, exploration strategies, and state-space complexity.

Starting from a basic Q-learning agent with poor performance, the project incrementally introduces methodological improvements and evaluates their impact on learning outcomes. The goal is not only to win games, but to understand **why certain RL design choices succeed or fail** in practice.

The project emphasizes:
- Reward signal engineering
- Exploration–exploitation tradeoffs
- Bias and variance in Q-learning
- State-space reduction via abstraction
- Empirical evaluation across multiple training runs

---
## From Coursework to Applied Reinforcement Learning
Developed for **ECS 170 (Artificial Intelligence), this project extends an initial Mancala reinforcement learning implementation adapted from [this repository](https://github.com/mkgray/mancala-reinforcement-learning).

### Initial Scaffolding
- Terminal-based Mancala game engine
- Support for 0, 1, or 2 human players
- Pretrained Q-learning agent
- Training against a **random action-selection opponent**
- 100,000-game baseline training run

### Original Baseline Model
- Tabular Q-learning
- Fixed epsilon-greedy exploration
- Sparse reward signal
- Evaluation metric:
  - Win = 1
  - Draw = 0
  - Loss = -1

#### Baseline Performance
- ~30% average win rate
- No convergence toward strategic play
- High variance across runs

This baseline served as a control for all subsequent experiments.

---
## Key Findings

### Learning Dynamics
- Reward density is the single most important factor in early learning
- Exploration scheduling (epsilon decay) improves stability and convergence
- Naive reward shaping can degrade performance if poorly aligned
- State abstraction dramatically outperforms algorithmic complexity alone

### Exploration & Stability
- Fixed epsilon-greedy exploration leads to premature convergence
- Decaying epsilon avoids suboptimal early policies
- Softmax exploration improves action selection quality but increases computational cost
- Hybrid strategies balance efficiency and performance

### Performance Summary

| Model Variant                              | Avg Win Rate |
|-------------------------------------------|--------------|
| Baseline Q-learning                       | ~30%         |
| Reward-adjusted baseline                  | 47.52%       |
| Potential-based shaping                  | 35.04%       |
| Epsilon decay                             | 49.66%       |
| Alpha decay                               | 43.30%       |
| Softmax + ε-greedy                        | 45.22%       |
| Double Q-learning                        | 46.29%       |
| **State aggregation + hybrid exploration** | **72.59%**   |

---

## Technical Implementation

### Core Algorithm: Q-Learning
The agent uses temporal difference Q-learning:
```
Q(s, a) ← Q(s, a) + α [ r + γ max Q(s′, a′) − Q(s, a) ]
```

Where:
- `s, s'` are board states
- `a` is a legal move
- `r` is the shaped reward
- `α` is the learning rate
- `γ` is the discount factor

---

## Key Technical Decisions

### 1. Reward Engineering

#### Initial Reward System
- Reward = total marbles at game end
- Resulted in sparse feedback and poor learning

#### Revised Reward System
- +1000 for winning
- -1000 for losing
- +5 per marble captured
- +2 per marble gained at end of turn

#### Result:
- Win rate increased to **47.52%**

### 2. Potential-Based Reward Shaping
Reward shaping introduced heuristic-based transition rewards:
- Increasing agent vs opponent store difference
- Adding stones to agent store
- Removing stones from opponent pockets
- Gaining extra turns
- Leaving opponent pockets empty

#### Outcome:
- Average win rate decreased to **35.04%**
- Excessive shaping introduced noisy learning signals

### 3. Exploration Scheduling (Epsilon Decay)
- Initial ε = 1.0
- Minimum ε = 0.05
- Gradual decay over training

#### Result:
- Average win rate improved to **49.66%**
- Reduced premature convergence

### 4. Learning Rate Scheduling (Alpha Decay)
- Initial α = 0.4
- Minimum α = 0.01
- Decay factor =~0.999

#### Result:
-  Average win rate: **43.3%**
-  Improved convergence, reduced adaptability

### 5. Action Selection Strategies

#### Softmax Exploration
- Probabilistic action selection based on Q-values
- Controlled by temperature parameter
- Computationally expensive

#### Hybrid Softmax + Epsilon-Greedy
- Random action with probability ε
- Softmax otherwise
- Reduced computational cost

#### Result:
- Win rate: **45.22%**
- No significant improvement over epsilon decay alone

### 6. Bias Reduction: Double Q-Learning
- Introduced a second Q-table
- Randomly selected estimator per update

#### Result:
- Win rate: **46.29%**
- Zero initialization limited early exploration

### 7. State Aggregation (Most Impactful Improvement)
To reduce state-space complexity:
- Grouped board configurations into meta-states
- Based on Mancala-specific heuristics (Divilly et al., 2013)
- Priority ordering: H5 → H4 → H2 → H3

#### Example Heuristics
- Maximum stones on own side
- Stone advantage over opponent
- Rightmost pocket availability
- Empty opponent pockets
- Distance of stones from goal

#### Results:
- State aggregation only: ~60% win rate
- With softmax + decay: ~72% win rate

### 8. Reward Scaling Experiments

#### Reward Clipping
- Clipped rewards to [−1, 1]
- Reduced win rate by ~4%
- Suppressed meaningful action differentiation

#### Reward Normalization
- Scaled rewards proportionally
- Slowed convergence
- Reduced Q-value contrast

Both approaches were rejected.

---

## Methodology

The project follows a structured experimental pipeline:

### 1. Environment Construction
  - Full Mancala rule engine
  - Legal action masking
  - Terminal-based interaction
### 2. Baseline Training
  - Q-learning against random opponent
  - Fixed exploration
### 3. Ablation Studies
  - Reward shaping
  - Exploration decay
  - Learning rate decay
  - Action selection strategies
### 4. State-Space Reduction
  - Heuristic-based aggregation
  - Meta-state learning
### 5. Evaluation
  - Multiple independent runs
  - Win-rate tracking
  - Moving average smoothing
  - Variance analysis

---

## Reproducibility

### Environment
- Python 3.6+
- NumPy
- Matplotlib

### Run
```
pip install numpy matplotlib
python train_mancala_agent.py
python play_mancala.py
```

Outputs include:
- Trained agent models
- Moving average plots
- Cross-run consistency analysis

---

## Project Structure
```
mancala-reinforcement-learning/
│
├── mancala/
│   ├── mancala.py              # Game engine
│   ├── agent.py                # Q-learning agent
│   ├── train_mancala_agent.py  # Training & analysis
│   ├── play_mancala.py         # Interactive gameplay
│   └── tests/
│   │   └── mancala_tests.py    # Script for tests
├── figures/                    # Moving average, win rate, & consistency plots
│   ├── Run1MovingAvg&WinRate.png
│   ├── Run2MovingAvg&WinRate.png
│   ├── Run3MovingAvg&WinRate.png
│   ├── Run4MovingAvg&WinRate.png
│   ├── Run5MovingAvg&WinRate.png
│   └── consistency_plot.png
├── CODE_OF_CONDUCT.md          # Contributor Code of Conduct
├── ECS 170 Project Presentation  # Project Presentation
├── README.md
└── LICENSE
```

---

## Ethical & Practical Considerations
- Training against random agents limits realism
- Learned policies may not reflect human strategies
- Persistent exploration affects human gameplay
- Results depend heavily on reward design choices

---

## Technologies Used
- Python 3.6+
- NumPy
- Matplotlib
- Pickle (model persistence)

### Reinforcement Learning Concepts
- Q-learning
- Exploration-exploitation tradeoffs
- Reward shaping
- State abstraction
- Bias-variance tradeoffs
- Convergence analysis

---

## Author
Developed by an 8-person team for ECS 170.

**Angelina Cottone, Nick Gomez, Harris Habib, Mythri Kulkarni, Aatish Lobo, Andrew Ortega, Shriya Rudrashetty, Alyssa Ann Toledo**
UC Davis, 2024

---
*Last Updated: January 2026*
