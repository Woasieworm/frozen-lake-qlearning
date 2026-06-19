"""
environment.py
FrozenLakeEnv: Custom 8x8 Frozen Lake environment from first principles.
No Gymnasium, OpenAI Gym, Stable Baselines, or any RL framework is used.

Actions : 0=Left  1=Down  2=Right  3=Up
States  : Single integer index  s = row * ncols + col,  range [0, 63]
Rewards : +1.0 reaching Goal | -1.0 falling in Hole | -0.01 each safe step
          (step penalty encourages efficient paths and provides a gradient
          signal in this sparse-reward environment)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Standard 8×8 Frozen Lake map
# ---------------------------------------------------------------------------
FROZEN_LAKE_MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

# Action constants
LEFT  = 0
DOWN  = 1
RIGHT = 2
UP    = 3

ACTION_SYMBOLS = {LEFT: "←", DOWN: "↓", RIGHT: "→", UP: "↑"}
ACTION_DELTAS  = {LEFT: (0, -1), DOWN: (1, 0), RIGHT: (0, 1), UP: (-1, 0)}

# Reward constants
REWARD_GOAL = 1.0
REWARD_HOLE = -1.0
REWARD_STEP = -0.01   # small step penalty → non-zero gradient before goal is found


class FrozenLakeEnv:
    """
    Custom Frozen Lake grid-world for 8×8 maps.

    State representation
    --------------------
    Single integer index:  s = row * ncols + col
    Terminal states       : holes (H) and the goal (G)

    Parameters
    ----------
    lake_map   : list[str] – grid rows; defaults to the standard 8×8 map.
    stochastic : bool      – Bonus Option A: actions have a slip_prob chance
                             of moving in a random direction instead.
    slip_prob  : float     – slip probability used when stochastic=True.
    """

    def __init__(self, lake_map=None, stochastic=False, slip_prob=0.1):
        self.lake_map   = lake_map if lake_map is not None else FROZEN_LAKE_MAP
        self.nrows      = len(self.lake_map)
        self.ncols      = len(self.lake_map[0])
        self.n_states   = self.nrows * self.ncols
        self.n_actions  = 4
        self.stochastic = stochastic
        self.slip_prob  = slip_prob

        self._parse_map()
        self.state = self.start_state

    # ------------------------------------------------------------------
    # Map parsing
    # ------------------------------------------------------------------

    def _parse_map(self):
        self.hole_states = set()
        self.goal_state  = None
        self.start_state = None
        for r, row in enumerate(self.lake_map):
            for c, cell in enumerate(row):
                s = self._rc_to_s(r, c)
                if   cell == "H": self.hole_states.add(s)
                elif cell == "G": self.goal_state  = s
                elif cell == "S": self.start_state = s
        assert self.start_state is not None, "No start state 'S' in map."
        assert self.goal_state  is not None, "No goal state 'G' in map."

    def _rc_to_s(self, row, col): return row * self.ncols + col
    def _s_to_rc(self, s):        return divmod(s, self.ncols)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset agent to start state. Returns the initial state index."""
        self.state = self.start_state
        return self.state

    def step(self, action):
        """
        Execute action from current state.

        Returns
        -------
        next_state : int
        reward     : float
        done       : bool
        info       : dict  – {'outcome': 'goal' | 'hole' | 'frozen'}
        """
        if self.is_terminal():
            raise RuntimeError("Episode is done. Call reset() before stepping.")

        if self.stochastic:
            action = self._stochastic_action(action)

        r, c   = self._s_to_rc(self.state)
        dr, dc = ACTION_DELTAS[action]

        nr = max(0, min(self.nrows - 1, r + dr))
        nc = max(0, min(self.ncols - 1, c + dc))

        next_state = self._rc_to_s(nr, nc)
        self.state = next_state

        if next_state == self.goal_state:
            return next_state, REWARD_GOAL, True,  {"outcome": "goal"}
        elif next_state in self.hole_states:
            return next_state, REWARD_HOLE, True,  {"outcome": "hole"}
        else:
            return next_state, REWARD_STEP, False, {"outcome": "frozen"}

    def get_state(self):
        """Return current state as integer index."""
        return self.state

    def is_terminal(self):
        """Return True if current state is a hole or the goal."""
        return self.state in self.hole_states or self.state == self.goal_state

    def render(self, mode="human"):
        """Print a text representation of the grid with agent marked as '*'."""
        ar, ac = self._s_to_rc(self.state)
        print()
        for r, row in enumerate(self.lake_map):
            line = "".join("*" if (r == ar and c == ac) else ch
                           for c, ch in enumerate(row))
            print(line)
        print()

    # ------------------------------------------------------------------
    # Stochastic transitions (Bonus Option A)
    # ------------------------------------------------------------------

    def _stochastic_action(self, intended):
        """With probability slip_prob, move in a random direction instead."""
        if np.random.random() < self.slip_prob:
            return np.random.randint(0, self.n_actions)
        return intended

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def cell_type(self, s):
        r, c = self._s_to_rc(s)
        return self.lake_map[r][c]

    def print_policy(self, policy):
        """Display the extracted policy as a symbol grid."""
        print("\n=== Learned Policy ===")
        for r in range(self.nrows):
            line = ""
            for c in range(self.ncols):
                s    = self._rc_to_s(r, c)
                cell = self.lake_map[r][c]
                if   cell == "H": line += " H "
                elif cell == "G": line += " G "
                else:
                    a = int(policy[s]) if hasattr(policy, "__getitem__") else policy(s)
                    line += f" {ACTION_SYMBOLS[a]} "
            print(line)
        print()
