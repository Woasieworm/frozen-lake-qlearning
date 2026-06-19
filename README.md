# Frozen Lake Q-Learning

**DSCD 614 - Reinforcement Learning | Programming Assignment 1**  
**University of Ghana, Department of Computer Science**  
**Semester II, 2025/2026 Academic Year**

---

## Introduction

### What is Reinforcement Learning?

Reinforcement Learning is a branch of machine learning where an agent learns to make decisions by interacting with an environment. At each step the agent observes the current state, picks an action, and receives a reward. Over time it learns a policy, which is a rule that tells it what to do in any given state, such that the total reward it collects is as high as possible. Unlike supervised learning, the agent is not given correct answers. It figures things out through trial and error.

### What is Frozen Lake?

Frozen Lake is a grid-world navigation problem. The agent starts at the top-left corner of an 8x8 grid and must reach the goal at the bottom-right corner without falling into any holes. Most cells are safe frozen ice, but ten cells are holes that end the episode immediately. It is a sparse-reward problem because the agent only receives a meaningful reward when it reaches the goal or falls in a hole, which makes it challenging to learn from.

The map used in this assignment is the standard 8x8 layout:

```
S F F F F F F F
F F F F F F F F
F F F H F F F F
F F F H F F F F
F F F H F F F F
F H H F F F H F
F H F F H F H F
F F F H F F F G
```

---

## Environment Design

### State Representation

Each cell in the grid is represented as a single integer using the formula:

```
state = row x 8 + col     (range: 0 to 63)
```

The start cell (row 0, col 0) is state 0. The goal cell (row 7, col 7) is state 63. The ten hole states are: 19, 27, 35, 41, 42, 46, 49, 52, 54, and 59.

### Action Representation

| Code | Direction | Grid Change      |
|------|-----------|------------------|
| 0    | Left      | (row, col - 1)   |
| 1    | Down      | (row + 1, col)   |
| 2    | Right     | (row, col + 1)   |
| 3    | Up        | (row - 1, col)   |

If an action would take the agent off the edge of the grid, it stays in its current cell.

### Reward Structure

| Event                  | Reward  |
|------------------------|---------|
| Reached the Goal (G)   | +1.0    |
| Fell in a Hole (H)     | -1.0    |
| Moved on safe ice (F)  | -0.01   |

The -0.01 step penalty was added to solve the sparse-reward problem. Without it, every non-terminal step returns zero and the Q-table has no signal to learn from until the agent happens to find the goal, which only occurs about 0.3% of the time with a random policy on this map. The step penalty gives every transition a meaningful value so Q-values can start propagating from the goal back through the grid from the very first episode.

---

## Q-Learning Algorithm

### Description

Q-Learning is a model-free, off-policy reinforcement learning algorithm. It maintains a Q-table where each entry Q(s, a) holds the agent's estimate of how much total future reward it can expect by taking action a in state s and then acting optimally from that point on. The agent does not need to know how the environment works in advance. It learns purely from experience.

### Update Equation

After every transition the Q-table is updated using the following equation, implemented exactly as specified in the assignment:

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
```

- `alpha` is the learning rate, controlling how much the agent revises its estimates
- `gamma` is the discount factor, controlling how much it values future rewards
- The bracketed term is the TD error, the difference between what was expected and what was received
- When the episode ends (done = True), the future value is set to zero

### Exploration Strategy

The agent uses epsilon-greedy action selection. With probability epsilon it picks a random action to explore, and otherwise it picks the action with the highest Q-value to exploit what it has learned. Epsilon is decayed after every episode:

```
epsilon <- max(epsilon_min, epsilon x decay_factor)
```

**Decaying epsilon-greedy** (primary strategy): epsilon starts at 1.0 and decays to 0.01 over 50,000 episodes, giving the agent a long exploration phase before exploitation takes over.

**Pure epsilon-greedy** (comparison strategy for Bonus C): epsilon is fixed at 0.1 throughout training, maintaining a constant mix of exploration and exploitation.

---

## Training Procedure

### Hyperparameters

| Parameter            | Value     | Reasoning                                          |
|----------------------|-----------|----------------------------------------------------|
| Learning rate        | 0.3       | Converges faster than 0.1 while staying stable     |
| Discount factor      | 0.99      | Places high value on reaching the distant goal     |
| Starting epsilon     | 1.0       | Agent starts by exploring fully at random          |
| Minimum epsilon      | 0.01      | Keeps a small amount of exploration after training |
| Epsilon decay        | 0.9999079 | Epsilon reaches minimum around episode 50,000      |
| Episodes             | 50,000    | Gives Q-values enough time to propagate fully      |
| Max steps per episode| 500       | Long enough to reach the goal from any position    |

### Number of Episodes

50,000 episodes were used. A greedy evaluation was run every 5,000 episodes using 200 test episodes with epsilon set to zero. The agent reached 100% greedy success rate by the first checkpoint at episode 5,000 and held it through episode 50,000.

---

## Results

### Final Success Rate

| Metric              | Value    |
|---------------------|----------|
| Episodes evaluated  | 500      |
| Successful runs     | 500      |
| Failures            | 0        |
| Success rate        | 100.00%  |
| Average reward      | 0.87     |

### Learned Policy

```
 down  down  down  down  down  down  down  down
 right right right right down  down  down  down
 right right up    H     down  down  down  down
 right right up    H     down  down  down  down
 right right up    H     right right right down
 up    H     H     right right up    H     down
 up    H     right up    H     up    H     down
 up    left  down  H     right up    right G
```

The agent moves right and downward through the top of the grid to reach column 7, then travels straight down that column to the goal. This path avoids all ten holes and is followed successfully in every evaluation episode.

### Discussion

The main challenge in this problem is the sparse reward. A random agent only reaches the goal about 0.3% of the time on this 8x8 map. Without the step penalty, the Q-table stays flat for thousands of episodes because there is no signal to learn from. Adding -0.01 per safe step fixed this by creating a gradient throughout the state space from episode one.

The epsilon decay schedule was equally important. A fast decay (such as 0.995) collapses epsilon below 0.01 before the Q-table has built up any useful values, locking the agent into a bad policy permanently. Keeping epsilon above 0.3 for the first 15,000 episodes gave the Q-table enough time to become reliable before exploitation took over.

Decaying epsilon-greedy is preferred over pure epsilon-greedy because it ends with a fully deterministic policy. Pure epsilon-greedy permanently retains 10% random decisions even after the agent has fully converged.

---

## Execution Instructions

### Requirements

```
Python 3.9 or higher
numpy >= 1.24.0
matplotlib >= 3.7.0
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the agent

```bash
python train.py
```

This trains the agent for 50,000 episodes, prints greedy evaluation checkpoints every 5,000 episodes, saves the Q-table to `results/q_table.npy`, saves the training plot to `results/training_curves.png`, displays the learned policy, and runs the Bonus C exploration comparison saving `results/exploration_comparison.png`.

Training takes roughly 3 to 5 minutes depending on your machine.

### Evaluate the trained agent

```bash
python evaluate.py
```

This loads the saved Q-table and runs 500 greedy episodes with epsilon set to zero. It prints the success rate, average reward, and policy grid, and saves `results/evaluation_results.png`. Run this after train.py has completed.

---

## Repository Structure

```
frozen-lake-qlearning/
├── environment.py       # FrozenLakeEnv class (Part A)
├── agent.py             # QLearningAgent class (Part B)
├── train.py             # Training loop, policy extraction, Bonus B and C (Parts C and D)
├── evaluate.py          # Evaluation script (Part E)
├── requirements.txt
├── README.md
├── report.pdf
└── results/
    ├── q_table.npy
    ├── training_curves.png
    ├── evaluation_results.png
    └── exploration_comparison.png
```

---

## Bonus Features Implemented

| Option   | Description                                                                                     |
|----------|-------------------------------------------------------------------------------------------------|
| Option A | Stochastic transitions added to FrozenLakeEnv. Enable with `FrozenLakeEnv(stochastic=True)`    |
| Option B | 4-panel training dashboard saved to `results/training_curves.png`                              |
| Option C | Pure epsilon-greedy vs decaying epsilon-greedy comparison saved to `results/exploration_comparison.png` |
