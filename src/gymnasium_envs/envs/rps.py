import gymnasium as gym
from gymnasium import spaces

class RPSEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, opponent_p=(1, 0, 0)):
        self._opponent_p = opponent_p
        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(1)
        
    @staticmethod
    def _winner(player_draw, opponent_draw):
        # for 0,1,2 draws the following gives 1,0,-1 depending on win, tie, loss.
        result = player_draw - opponent_draw
        result %= 3
        result -= 1
        return result

    def reset(self,*, seed=None, options=None):
        super().reset(seed=seed)
        observation = 0
        info = {}
        return observation, info
    
    def step(self, action):
        opponent_draw = self.np_random.choice(3, p=self._opponent_p)
        observation = 0
        reward = self._winner(action, opponent_draw)
        info = {}
        terminated = True
        truncated = False
        return observation, reward, terminated, truncated, info
    
