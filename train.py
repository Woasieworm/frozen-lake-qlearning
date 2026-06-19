"""
train.py
Training loop for the Frozen Lake Q-Learning agent.

Reward shaping
--------------
The Frozen Lake environment is a *sparse-reward* problem: without shaping,
the agent only receives a signal when it accidentally reaches the goal during
random exploration (≈ 0.3% of episodes with max_steps=500 and a random policy).
To provide a non-zero learning gradient from the first episode, we use:
    +1.0  reaching the Goal
    -1.0  falling into a Hole
    -0.01 each safe step (step penalty)
The step penalty does not affect the optimality of the final policy because the
discounted sum still favours reaching the goal over any sequence of step penalties.

Bonus Option B – training curves visualised with matplotlib.
Bonus Option C – Pure ε-Greedy vs. Decaying ε-Greedy comparison.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import FrozenLakeEnv
from agent import QLearningAgent


# ======================================================================
# Hyperparameters
# ======================================================================
# epsilon_decay = 0.9999079  →  ε ≈ 0.63 at ep 5k / 0.40 at ep 10k
#                               / 0.16 at ep 20k / 0.01 at ep 50k
# This ensures ample exploration while goal-discovery propagates backwards.
HYPERPARAMS = {
    "alpha":         0.3,
    "gamma":         0.99,
    "epsilon":       1.0,
    "epsilon_min":   0.01,
    "epsilon_decay": 0.9999079,   # reaches 0.01 by episode ~50 000
    "n_episodes":    50_000,
    "max_steps":     500,
    "eval_interval": 5_000,
    "eval_episodes": 200,
}

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ======================================================================
# Greedy checkpoint evaluation
# ======================================================================

def greedy_eval(env, agent, n_episodes=200, max_steps=500):
    """Run n_episodes with ε=0 and return success rate (0–100)."""
    eval_env = FrozenLakeEnv()          # separate env – never shares state
    saved    = agent.epsilon
    agent.epsilon = 0.0
    successes = 0
    for _ in range(n_episodes):
        s = eval_env.reset()
        for _ in range(max_steps):
            a              = agent.select_action(s, greedy=True)
            ns, _, done, info = eval_env.step(a)
            s              = ns
            if done:
                if info["outcome"] == "goal":
                    successes += 1
                break
    agent.epsilon = saved
    return 100 * successes / n_episodes


# ======================================================================
# Core training function
# ======================================================================

def train(env, agent, n_episodes=50_000, max_steps=500,
          eval_interval=5_000, eval_episodes=200):
    """
    Run the Q-Learning training loop.

    Returns
    -------
    stats : dict
        episode_rewards, success_flags, epsilon_history,
        rolling_success, eval_checkpoints
    """
    episode_rewards  = []
    success_flags    = []
    epsilon_history  = []
    eval_checkpoints = {}

    for ep in range(1, n_episodes + 1):
        state        = env.reset()
        total_reward = 0.0
        success      = False

        for _ in range(max_steps):
            action                         = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state        = next_state
            total_reward += reward
            if done:
                if info["outcome"] == "goal":
                    success = True
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        success_flags.append(int(success))
        epsilon_history.append(agent.epsilon)

        if ep % eval_interval == 0:
            rate = greedy_eval(env, agent,
                               n_episodes=eval_episodes, max_steps=max_steps)
            eval_checkpoints[ep] = rate
            recent_wins = sum(success_flags[-eval_interval:])
            print(
                f"Episode {ep:>6}/{n_episodes}  │  "
                f"Train wins (last {eval_interval}): {recent_wins:>4}  │  "
                f"Greedy SR: {rate:5.1f}%  │  "
                f"ε: {agent.epsilon:.4f}"
            )

    window_size     = 500
    rolling_success = [
        100 * np.mean(success_flags[max(0, i - window_size): i + 1])
        for i in range(len(success_flags))
    ]

    return {
        "episode_rewards":  episode_rewards,
        "success_flags":    success_flags,
        "epsilon_history":  epsilon_history,
        "rolling_success":  rolling_success,
        "eval_checkpoints": eval_checkpoints,
    }


# ======================================================================
# Plotting  (Bonus Option B)
# ======================================================================

def plot_training(stats, title_suffix="", save_path=None):
    """Four-panel training dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Q-Learning Training – Frozen Lake 8×8  {title_suffix}",
        fontsize=14, fontweight="bold"
    )

    n   = len(stats["episode_rewards"])
    eps = range(1, n + 1)

    # Panel 1: episode rewards
    axes[0, 0].plot(eps, stats["episode_rewards"],
                    alpha=0.2, color="steelblue", lw=0.3)
    # Smooth with 500-ep moving average
    smooth = np.convolve(stats["episode_rewards"],
                         np.ones(500) / 500, mode="valid")
    axes[0, 0].plot(range(500, n + 1), smooth, color="navy", lw=1.2, label="MA-500")
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].legend()

    # Panel 2: rolling success + greedy checkpoints
    axes[0, 1].plot(eps, stats["rolling_success"],
                    color="seagreen", lw=1.2, label="Rolling (window=500)")
    ck = stats.get("eval_checkpoints", {})
    if ck:
        ck_x = [int(k) for k in ck]
        ck_y = [ck[k] for k in ck]
        axes[0, 1].plot(ck_x, ck_y, "o--", color="purple", lw=1.4,
                        markersize=6, label=f"Greedy eval ({list(ck.values())[-1]:.0f}%)")
    axes[0, 1].axhline(80, color="red", ls="--", lw=0.9, label="80% threshold")
    axes[0, 1].set_title("Success Rate")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Success Rate (%)")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].legend()

    # Panel 3: epsilon decay
    axes[1, 0].plot(eps, stats["epsilon_history"], color="darkorange", lw=1.2)
    axes[1, 0].set_title("Exploration Rate (ε) Decay")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Epsilon")

    # Panel 4: cumulative successes
    cumulative = np.cumsum(stats["success_flags"])
    axes[1, 1].plot(eps, cumulative, color="teal", lw=1.2)
    axes[1, 1].set_title("Cumulative Successful Episodes")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Cumulative Successes")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training plot saved → {save_path}")
    plt.close()


# ======================================================================
# Bonus Option C – compare Pure vs Decaying ε-Greedy
# ======================================================================

def compare_exploration_strategies(n_episodes=50_000, max_steps=500,
                                   eval_interval=5_000, eval_episodes=200):
    """
    Train two agents side-by-side and compare their greedy evaluation curves:
      • Pure ε-Greedy     – ε fixed at 0.1 throughout
      • Decaying ε-Greedy – ε starts at 1.0 and decays to 0.01 over 50 000 episodes
    """
    print("\n" + "=" * 55)
    print("  Bonus Option C: Exploration Strategy Comparison")
    print("=" * 55)

    strategies = {
        "Pure ε-Greedy (ε=0.1)": {
            "epsilon": 0.1, "epsilon_min": 0.1, "epsilon_decay": 1.0,
        },
        "Decaying ε-Greedy (1.0 → 0.01)": {
            "epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.9999079,
        },
    }

    all_results = {}
    for name, cfg in strategies.items():
        env   = FrozenLakeEnv()
        agent = QLearningAgent(
            n_states=env.n_states, n_actions=env.n_actions,
            alpha=HYPERPARAMS["alpha"], gamma=HYPERPARAMS["gamma"], **cfg,
        )
        print(f"\nTraining: {name}")
        stats = train(env, agent, n_episodes=n_episodes,
                      max_steps=max_steps, eval_interval=eval_interval,
                      eval_episodes=eval_episodes)
        all_results[name] = stats

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exploration Strategy Comparison – Frozen Lake 8×8",
                 fontsize=13, fontweight="bold")

    colours = {
        "Pure ε-Greedy (ε=0.1)":             "tomato",
        "Decaying ε-Greedy (1.0 → 0.01)":    "steelblue",
    }

    for name, stats in all_results.items():
        ck  = stats["eval_checkpoints"]
        x   = [int(k) for k in ck]
        y   = [ck[k]  for k in ck]
        axes[0].plot(x, y, "o-", label=name, color=colours[name], lw=1.5, ms=5)
        axes[1].plot(stats["epsilon_history"], label=name,
                     color=colours[name], lw=1.2, alpha=0.85)

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Greedy Success Rate (%)")
    axes[0].set_title("Greedy Evaluation Checkpoints")
    axes[0].legend()
    axes[0].set_ylim(0, 105)
    axes[0].axhline(80, color="grey", ls="--", lw=0.8)

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Epsilon")
    axes[1].set_title("Exploration Rate Over Training")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exploration_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nComparison plot saved → {path}")
    return all_results


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("  DSCD 614 – RL Assignment 1: Frozen Lake Q-Learning")
    print("=" * 60)
    print(f"\nHyperparameters:\n")
    for k, v in HYPERPARAMS.items():
        print(f"  {k:<18}: {v}")
    print()

    env   = FrozenLakeEnv()
    agent = QLearningAgent(
        n_states      = env.n_states,
        n_actions     = env.n_actions,
        alpha         = HYPERPARAMS["alpha"],
        gamma         = HYPERPARAMS["gamma"],
        epsilon       = HYPERPARAMS["epsilon"],
        epsilon_min   = HYPERPARAMS["epsilon_min"],
        epsilon_decay = HYPERPARAMS["epsilon_decay"],
    )

    print(f"Agent : {agent}")
    print(f"Grid  : {env.nrows}×{env.ncols} | States: {env.n_states} | Actions: {env.n_actions}\n")

    print("Training…\n")
    stats = train(
        env, agent,
        n_episodes    = HYPERPARAMS["n_episodes"],
        max_steps     = HYPERPARAMS["max_steps"],
        eval_interval = HYPERPARAMS["eval_interval"],
        eval_episodes = HYPERPARAMS["eval_episodes"],
    )

    # Final evaluation
    final_rate = greedy_eval(env, agent, n_episodes=500, max_steps=500)
    print(f"\nFinal greedy success rate (500 episodes): {final_rate:.1f}%")

    # Save Q-table
    q_path = os.path.join(RESULTS_DIR, "q_table.npy")
    agent.save_q_table(q_path)

    # Save stats
    stats_path = os.path.join(RESULTS_DIR, "train_stats.json")
    serial = {k: (v.tolist() if hasattr(v, "tolist") else v)
              for k, v in stats.items()}
    serial["final_greedy_success_rate"] = final_rate
    serial["hyperparams"] = HYPERPARAMS
    with open(stats_path, "w") as f:
        json.dump(serial, f, indent=2)
    print(f"Stats saved → {stats_path}")

    # Plots
    plot_training(stats, save_path=os.path.join(RESULTS_DIR, "training_curves.png"))

    # Policy
    env.print_policy(agent.get_policy())

    # Bonus C
    compare_exploration_strategies(n_episodes=50_000)

    print("\nAll results written to ./results/")


if __name__ == "__main__":
    main()
