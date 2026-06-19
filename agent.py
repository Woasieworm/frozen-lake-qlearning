"""
agent.py
Q-Learning agent implemented from first principles.

Update rule:
    Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
"""

import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning agent with epsilon-greedy exploration and
    decaying epsilon schedule.

    Parameters
    ----------
    n_states       : int   – total number of states.
    n_actions      : int   – total number of actions.
    alpha          : float – learning rate (α), default 0.1.
    gamma          : float – discount factor (γ), default 0.99.
    epsilon        : float – initial exploration rate, default 1.0.
    epsilon_min    : float – minimum epsilon after decay, default 0.01.
    epsilon_decay  : float – multiplicative decay applied after each episode.
    q_init         : float – initial Q-table value, default 0.0.
    """

    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        q_init=0.0,
    ):
        self.n_states      = n_states
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: shape (n_states, n_actions), initialised to q_init
        self.Q = np.full((n_states, n_actions), q_init, dtype=np.float64)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state, greedy=False):
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state   : int  – current state index.
        greedy  : bool – if True, always exploit (used during evaluation).

        Returns
        -------
        action : int
        """
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)     # explore
        return int(np.argmax(self.Q[state]))                 # exploit

    # ------------------------------------------------------------------
    # Q-table update
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state, done):
        """
        Apply the Q-Learning update equation.

        Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]

        When done=True the future value is 0 (terminal state).
        """
        best_next = 0.0 if done else float(np.max(self.Q[next_state]))
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """Multiplicatively decay epsilon, floored at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Policy extraction
    # ------------------------------------------------------------------

    def get_policy(self):
        """
        Extract the greedy policy from the Q-table.

        Returns
        -------
        policy : np.ndarray, shape (n_states,)
            policy[s] = argmax_a Q(s, a)
        """
        return np.argmax(self.Q, axis=1)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_q_table(self, path):
        """Save Q-table to a .npy file."""
        np.save(path, self.Q)
        print(f"Q-table saved → {path}")

    def load_q_table(self, path):
        """Load Q-table from a .npy file."""
        self.Q = np.load(path)
        print(f"Q-table loaded ← {path}")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"QLearningAgent(α={self.alpha}, γ={self.gamma}, "
            f"ε={self.epsilon:.4f}, ε_min={self.epsilon_min}, "
            f"ε_decay={self.epsilon_decay})"
        )
