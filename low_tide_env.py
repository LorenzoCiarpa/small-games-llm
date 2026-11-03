"""
Low Tide - a small grid-world environment for reinforcement learning.

The agent is a shell collector on a beach: a 10x10 grid where row 0 is the
land side (top) and row 9 is the sea side (bottom). The tide rises and falls,
and the water covers the bottom `level` rows of the grid.

  * Every time the tide RECEDES by one row, the row it uncovers is littered
    with shells and jellyfish.
  * Every time the tide RISES by one row, everything on the newly flooded row
    (shells and jellyfish) is washed away.
  * Collecting a shell (stepping on it) gives +1.
  * Stepping on a jellyfish gives -1, removes the jellyfish and paralyses the
    agent for STUN_STEPS steps (its actions are ignored).
  * The agent cannot walk into the water (a move into a water cell does
    nothing), but if the rising tide floods the row the agent is standing
    on, the agent drowns and the episode ends.

The tide never goes above level 7, so rows 0-2 are always dry (and never get
shells). Shells only ever appear in rows 3-9, the intertidal zone.

Observation: float32 vector of length OBS_DIM = 403
    obs[0:400]  four binary 10x10 planes, flattened in (channel, row, col) order:
                index = channel * 100 + row * 10 + col
                channel 0: agent      (exactly one cell is 1)
                channel 1: water      (all rows >= 10 - level are 1)
                channel 2: shell
                channel 3: jellyfish
    obs[400]    tide_rising: 1.0 if the NEXT tide change raises the water, else 0.0
    obs[401]    countdown:   (steps until the next tide change) / 5, in {0.2, 0.4, ..., 1.0}
                countdown == 0.2 means the tide changes at the end of the next step
    obs[402]    stun:        (remaining paralysis steps) / 3, in {0, 1/3, 2/3, 1}

Actions: 0 = noop, 1 = up (row - 1), 2 = down (row + 1),
         3 = left (col - 1), 4 = right (col + 1).
Moving outside the grid or into a water cell leaves the agent where it is.

Order of events inside step(action):
    1. Sticky actions: with probability sticky_prob the previous action is
       repeated instead of the chosen one.
    2. If the agent is paralysed (stun > 0), the action is replaced by noop
       and stun decreases by 1. Otherwise the agent moves (unless the target
       cell is outside the grid or under water).
    3. If the agent is on a shell: reward +1, the shell is removed.
       If the agent is on a jellyfish: reward -1, the jellyfish is removed,
       stun = STUN_STEPS.
    4. Tide update: countdown decreases by 1; when it reaches 0 the level
       changes by one row (+1 rising, -1 receding), the flooded row is washed
       or the uncovered row is populated, and a new countdown is drawn from
       TIDE_INTERVALS. When the level reaches its current target, the
       direction flips and a new target is drawn (from HIGH_TIDE_LEVELS when
       it starts rising, from LOW_TIDE_LEVELS when it starts receding).
    5. If the tide has flooded the agent's row, the agent drowns: the
       episode ends (no extra penalty, but no more shells either).
    6. The episode is truncated after max_steps steps.

Randomness: the tide schedule and the spawning of shells/jellyfish use their
own random generator (seeded by reset), so for a given seed the world evolves
in exactly the same way whatever the agent does. Sticky actions use a separate
generator.
"""

import numpy as np

# ---------------------------------------------------------------- constants
H, W = 10, 10
N_CHANNELS = 4
CH_AGENT, CH_WATER, CH_SHELL, CH_JELLY = 0, 1, 2, 3
N_SCALARS = 3
IDX_RISING = N_CHANNELS * H * W          # 400
IDX_COUNTDOWN = IDX_RISING + 1           # 401
IDX_STUN = IDX_RISING + 2                # 402
OBS_DIM = N_CHANNELS * H * W + N_SCALARS # 403

ACTION_NAMES = ("noop", "up", "down", "left", "right")
N_ACTIONS = 5
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))  # (d_row, d_col)

LOW_TIDE_LEVELS = (0, 1, 2)     # possible lowest levels of a cycle
HIGH_TIDE_LEVELS = (5, 6, 7)    # possible highest levels of a cycle
TIDE_INTERVALS = (3, 4, 5)      # steps between two consecutive level changes
MAX_INTERVAL = max(TIDE_INTERVALS)

P_SHELL = 0.25                  # per cell of an uncovered row
P_JELLY = 0.08                  # per cell of an uncovered row
STUN_STEPS = 3
SHELL_REWARD = 1.0
STING_REWARD = -1.0

MAX_STEPS = 1000
STICKY_PROB = 0.1


class LowTideEnv:
    def __init__(self, sticky_prob=STICKY_PROB, max_steps=MAX_STEPS, seed=None):
        self.sticky_prob = sticky_prob
        self.max_steps = max_steps
        self._seed = seed
        self.world_rng = np.random.default_rng(seed)
        self.sticky_rng = np.random.default_rng(None if seed is None else seed + 1_000_003)

    # ------------------------------------------------------------ helpers
    def _in_water(self, r):
        return r >= H - self.level

    def _spawn_row(self, r):
        u = self.world_rng.random(W)
        self.shell[r] = u < P_SHELL
        self.jelly[r] = (u >= P_SHELL) & (u < P_SHELL + P_JELLY)

    def _new_target(self):
        if self.direction > 0:
            return int(self.world_rng.choice(HIGH_TIDE_LEVELS))
        return int(self.world_rng.choice(LOW_TIDE_LEVELS))

    def _tide_step(self):
        self.countdown -= 1
        if self.countdown > 0:
            return
        old = self.level
        self.level += self.direction
        if self.direction > 0:
            r = H - self.level            # newly flooded row
            self.shell[r] = False
            self.jelly[r] = False
        else:
            self._spawn_row(H - old)      # newly uncovered row
        if self.level == self.target:
            self.direction = -self.direction
            self.target = self._new_target()
        self.countdown = int(self.world_rng.choice(TIDE_INTERVALS))

    def _obs(self):
        grid = np.zeros((N_CHANNELS, H, W), dtype=np.float32)
        grid[CH_AGENT, self.agent_r, self.agent_c] = 1.0
        if self.level > 0:
            grid[CH_WATER, H - self.level:, :] = 1.0
        grid[CH_SHELL] = self.shell
        grid[CH_JELLY] = self.jelly
        scalars = np.array(
            [1.0 if self.direction > 0 else 0.0,
             self.countdown / MAX_INTERVAL,
             self.stun / STUN_STEPS],
            dtype=np.float32,
        )
        return np.concatenate([grid.ravel(), scalars])

    def _info(self):
        return {"t": self.t, "level": self.level, "shells": self.shells_collected,
                "stings": self.stings, "drowned": self.drowned}

    # ------------------------------------------------------------ API
    def reset(self, seed=None):
        if seed is not None:
            self.world_rng = np.random.default_rng(seed)
            self.sticky_rng = np.random.default_rng(seed + 1_000_003)
        self.shell = np.zeros((H, W), dtype=bool)
        self.jelly = np.zeros((H, W), dtype=bool)

        # tide: start somewhere in the middle of a cycle
        self.level = int(self.world_rng.integers(2, 6))            # 2..5
        self.direction = int(self.world_rng.choice((-1, 1)))
        if self.direction > 0:
            self.target = int(self.world_rng.choice([l for l in HIGH_TIDE_LEVELS if l > self.level]))
        else:
            self.target = int(self.world_rng.choice([l for l in LOW_TIDE_LEVELS if l < self.level]))
        self.countdown = int(self.world_rng.choice(TIDE_INTERVALS))

        # the dry part of the intertidal zone starts already populated
        for r in range(H - max(HIGH_TIDE_LEVELS), H - self.level):
            self._spawn_row(r)

        self.agent_r = 1
        self.agent_c = int(self.world_rng.integers(0, W))
        self.stun = 0
        self.last_action = 0
        self.t = 0
        self.shells_collected = 0
        self.stings = 0
        self.drowned = False
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"invalid action {action}")
        if self.sticky_rng.random() < self.sticky_prob:
            action = self.last_action
        self.last_action = action

        reward = 0.0
        if self.stun > 0:
            self.stun -= 1
        else:
            dr, dc = MOVES[action]
            nr, nc = self.agent_r + dr, self.agent_c + dc
            if 0 <= nr < H and 0 <= nc < W and not self._in_water(nr):
                self.agent_r, self.agent_c = nr, nc

        self.t += 1

        r, c = self.agent_r, self.agent_c
        if self.shell[r, c]:
            self.shell[r, c] = False
            reward += SHELL_REWARD
            self.shells_collected += 1
        elif self.jelly[r, c]:
            self.jelly[r, c] = False
            reward += STING_REWARD
            self.stings += 1
            self.stun = STUN_STEPS

        self._tide_step()
        if self._in_water(self.agent_r):
            self.drowned = True
            return self._obs(), reward, True, False, self._info()

        truncated = self.t >= self.max_steps
        return self._obs(), reward, False, truncated, self._info()

    def render(self):
        """ASCII view: A agent, o shell, j jellyfish, ~ water, . sand."""
        lines = []
        for r in range(H):
            row = []
            for c in range(W):
                if r == self.agent_r and c == self.agent_c:
                    row.append("A")
                elif self._in_water(r):
                    row.append("~")
                elif self.shell[r, c]:
                    row.append("o")
                elif self.jelly[r, c]:
                    row.append("j")
                else:
                    row.append(".")
            lines.append(" ".join(row))
        arrow = "rising" if self.direction > 0 else "receding"
        lines.append(f"t={self.t} level={self.level} ({arrow}, change in {self.countdown}) "
                     f"stun={self.stun} shells={self.shells_collected} stings={self.stings}")
        return "\n".join(lines)


if __name__ == "__main__":
    env = LowTideEnv()
    obs, info = env.reset(seed=0)
    print(env.render())
    total = 0.0
    rng = np.random.default_rng(0)
    while True:
        obs, r, term, trunc, info = env.step(rng.integers(N_ACTIONS))
        total += r
        if term or trunc:
            break
    print(env.render())
    print("random policy return:", total, info)
