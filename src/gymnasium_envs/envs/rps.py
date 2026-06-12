from abc import abstractmethod, ABC
from collections import deque
from typing import Type

import gymnasium as gym
from gymnasium import spaces
import numpy as np

#sentinel for "no previous moves"
NO_MOVE = 3

class Opponent(ABC):
    name: str # Will be set by the registry
    @abstractmethod
    def move(self, state, np_random) -> int:
        pass

class OpponentRegistry:

    def __init__(self):
        self.registry: dict[str, type[Opponent]] = {}
    
    def register(self, name: str):
        def decorator(opponent: Type[Opponent]):
            self.registry[name] = opponent
            return opponent
        
        return decorator
        
    def __call__(self, name: str):
        return self.registry[name]
    
    @property
    def list(self) -> list[str]:
        return list(self.registry.keys())
    
    def make_opponents(self, names: list[str] | None = None) -> list[Opponent]:
        if names is None:
            names = self.list
        opponents = []
        for name in names:
            opponent = self(name)() #create an intsance
            opponent.name = name
            opponents.append(opponent)
            
        return opponents
        
opponent_registry = OpponentRegistry()

@opponent_registry.register("Random")
class RandomOpponent(Opponent):
    def move(self, state, np_random):
        return np_random.choice(3)

@opponent_registry.register("SameMove")
class OneMoveOpponent(Opponent):
    def move(self, state, np_random):
        return 0
    
@opponent_registry.register("EchoMove")
class EchoMoveOpponent(Opponent):
    def move(self, state, np_random):
        last = state["player_history"][0]   # row 0 = previous round, most-recent first
        if last == NO_MOVE:                 # first round, nothing to echo
            return int(np_random.choice(3))
        return int(last)

@opponent_registry.register("Cycle")
class CycleOpponent(Opponent):
    def move(self, state, np_random):
        return int(state["round"]) % 3
      

class RPSEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, 
                render_mode=None, 
                opponents: list[str] | None = None,
                history_len: int = 1,
                n_rounds: int = 10,
        ):
        self.n_rounds = n_rounds
        self.history_len = history_len
        self.opponents = opponent_registry.make_opponents(opponents)
        self._opponents_ids = {opp.name: i for i, opp in enumerate(self.opponents)}

        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode
        self.observation_space = spaces.Dict({
            "opponent": spaces.Discrete(len(self.opponents)),
        })
        # the complete system state (superset; e.g. also exposes the round counter)
        self.state_space = spaces.Dict({
            "opponent":         spaces.Discrete(len(self.opponents)),
            "player_history":   spaces.MultiDiscrete([4] * history_len),
            "opponent_history": spaces.MultiDiscrete([4] * history_len),
            "round":            spaces.Discrete(n_rounds + 1),
        })
        
    @staticmethod
    def _winner(player_draw, opponent_draw):
        # for 0,1,2 draws the following gives 1,0,-1 depending on win, tie, loss.
        result = player_draw - opponent_draw + 1
        result %= 3
        result -= 1
        return result

    def _get_full_state(self):
        player_hist   = np.array([p for p, _ in self._history], dtype=np.int64)
        opponent_hist = np.array([o for _, o in self._history], dtype=np.int64)
        return {
            "opponent":         self._opponent_idx,
            "player_history":   player_hist,
            "opponent_history": opponent_hist,
            "round":            self._round,
        }

    def _get_obs(self):
        state = self._get_full_state()
        return {k: state[k] for k in self.observation_space.spaces}


    def reset(self,*, seed=None, options=None):
        super().reset(seed=seed)
        self._round = 0
        self._history = deque(
            [(NO_MOVE, NO_MOVE)] * self.history_len
        )
        self._opponent_idx = int(self.np_random.integers(len(self.opponents)))
        self._opponent = self.opponents[self._opponent_idx]
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        opponent_move = self._opponent.move(
            self._get_full_state(), 
            self.np_random
            )
        reward = self._winner(action, opponent_move)
        self._history.appendleft((action, opponent_move))
        self._round += 1
        info = {}
        terminated = self._round >= self.n_rounds
        truncated = False
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
    
