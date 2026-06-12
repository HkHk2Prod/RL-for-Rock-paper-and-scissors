import gymnasium as gym
from typing import Any

class Trainer:
    def __init__(
            self,
            env: gym.Env, 
            agent: Any, # Need to define agent ABC
    ):
        self.agent = agent
        self.env = env
        self.obs, self.info = self.env.reset()


    def episode(self):
        self.obs, self.info = self.env.reset()
        done = False
        while not done:
            action = self.agent.get_action(self.obs)

            next_obs, reward, terminated, truncated, info =\
                  self.env.step(action)

            self.agent.update(
                self.obs,
                action, 
                reward, 
                terminated, 
                next_obs
            )

            done = terminated or truncated
            self.obs = next_obs
        
        self.agent.decay_epsilon()




