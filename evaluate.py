"""
evaluate.py
Evaluate a trained Q-Learning agent on the Frozen Lake environment.

Loads the saved Q-table from results/q_table.npy and runs 500 greedy episodes.
Reports: Success Rate (%), Average Reward, Failures, Successful Runs.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import FrozenLakeEnv
from agent import QLearningAgent

RESULTS_DIR  = "results"
Q_TABLE_PATH = os.path.join(RESULTS_DIR, "q_table.npy")


def evaluate(agent, n_episodes=500, max_steps=500, verbose=True):
    """
    Run greedy evaluation (ε=0) in a fresh environment.

    Returns
    -------
    summary : dict
    rewards : list[float]
    """
    eval_env  = FrozenLakeEnv()
    successes = 0
    failures  = 0
    rewards   = []

    for ep in range(1, n_episodes + 1):
        s            = eval_env.reset()
        total_reward = 0.0
        success      = False
        for _ in range(max_steps):
            a              = agent.select_action(s, greedy=True)
            ns, r, done, info = eval_env.step(a)
            s              = ns
            total_reward  += r
            if done:
                if info["outcome"] == "goal":
                    success = True
                break
        rewards.append(total_reward)
        if success: successes += 1
        else:        failures  += 1

    success_rate = 100 * successes / n_episodes
    avg_reward   = float(np.mean(rewards))

    summary = {
        "n_episodes":   n_episodes,
        "successes":    successes,
        "failures":     failures,
        "success_rate": success_rate,
        "avg_reward":   avg_reward,
    }

    if verbose:
        print("\n" + "=" * 48)
        print("            EVALUATION RESULTS")
        print("=" * 48)
        print(f"  Episodes evaluated  : {n_episodes}")
        print(f"  Successful runs     : {successes}")
        print(f"  Failures            : {failures}")
        print(f"  Success Rate        : {success_rate:.2f}%")
        print(f"  Average Reward      : {avg_reward:.4f}")
        print("=" * 48)

    return summary, rewards


def plot_evaluation(rewards, save_path=None):
    """Pie chart of outcomes + reward histogram."""
    successes = sum(1 for r in rewards if r > 0)
    failures  = len(rewards) - successes

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Evaluation Results – Frozen Lake Q-Learning (Greedy Policy)",
                 fontsize=13, fontweight="bold")

    # Outcome pie
    axes[0].pie(
        [successes, failures],
        labels=[f"Success\n({successes})", f"Failure\n({failures})"],
        colors=["seagreen", "tomato"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    axes[0].set_title("Outcome Distribution")

    # Reward histogram
    axes[1].hist(rewards, bins=30, color="steelblue", edgecolor="white", rwidth=0.85)
    axes[1].axvline(float(np.mean(rewards)), color="red", ls="--",
                    lw=1.2, label=f"Mean = {np.mean(rewards):.3f}")
    axes[1].set_xlabel("Episode Reward")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Reward Distribution")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Evaluation plot saved → {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("  DSCD 614 – RL Assignment 1: Evaluation")
    print("=" * 60)

    if not os.path.exists(Q_TABLE_PATH):
        print(f"\n[ERROR] Q-table not found at {Q_TABLE_PATH}.")
        print("Please run  python train.py  first.")
        return

    # Build agent (hyperparams don't matter for greedy eval; ε=0)
    env   = FrozenLakeEnv()
    agent = QLearningAgent(
        n_states=env.n_states, n_actions=env.n_actions, epsilon=0.0,
    )
    agent.load_q_table(Q_TABLE_PATH)

    # Evaluate
    summary, rewards = evaluate(agent, n_episodes=500, max_steps=500)

    # Display learned policy
    env.print_policy(agent.get_policy())

    # Plot
    plot_evaluation(rewards,
                    save_path=os.path.join(RESULTS_DIR, "evaluation_results.png"))

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
