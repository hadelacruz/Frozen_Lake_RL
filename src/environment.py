import gymnasium as gym
import numpy as np


class FrozenLakeEnv:
    
    def __init__(self, is_slippery=True, render_mode="rgb_array"):
        self.env = gym.make("FrozenLake-v1", is_slippery=is_slippery, render_mode=render_mode)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        
    def get_state_action_space(self):
        return self.n_states, self.n_actions
    
    def reset(self):
        state, _ = self.env.reset()
        return state
    
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
