

from collections import defaultdict
from typing import Any

import numpy as np
import gymnasium as gym

class RPSAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        #tracking training 
        self.training_error = []

    @staticmethod
    def _key(obs: Any):
        """Turn a (possibly Dict/array) observation into a hashable Q-table key."""
        if isinstance(obs, dict):
            return tuple((k, RPSAgent._key(obs[k])) for k in sorted(obs))
        if isinstance(obs, np.ndarray):
            return tuple(obs.tolist())
        return obs

    def get_action(self, obs: Any) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[self._key(obs)]))

    def update(
            self,
            obs: Any,
            action: int,
            reward: float,
            terminated: bool,
            next_obs: Any,
    ):
        obs_key = self._key(obs)
        next_key = self._key(next_obs)

        future_q_value = (not terminated) * np.max(self.q_values[next_key])

        target = reward + self.discount_factor * future_q_value

        temporal_diff = target - self.q_values[obs_key][action]

        self.q_values[obs_key][action] = self.q_values[obs_key][action] + self.lr * temporal_diff

        self.training_error.append(temporal_diff)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def __str__(self):
        result = ""
        for k, v in self.q_values.items():
            opponent_idx = dict(k)["opponent"]
            name = self.env.unwrapped.opponents[opponent_idx].name
            result += f"{name}: {v}\n"
        return result
        