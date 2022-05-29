import numpy as np
from typing import Tuple
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class Memory(object):
    def __init__(self, 
                 size: int, 
                 state_shape: Tuple, 
                 action_space: Tuple, 
                 device=torch.device("cpu")
                 ) -> None:
        self.states = torch.zeros(size, *state_shape).to(device)
        if action_space.__class__.__name__ == 'Discrete' or type(action_space) == int:
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(size, action_shape).to(device)
        self.action_log_probs = torch.zeros(size, action_shape).to(device)
        self.next_states = torch.zeros(size, *state_shape).to(device)
        self.rewards = torch.zeros(size).to(device)
        self.advantages = torch.zeros(size + 1).to(device)
        self.returns = torch.zeros(size + 1).to(device)
        self.value_preds = torch.zeros(size + 1).to(device)
        self.dones = torch.zeros(size).to(device)
        
        self.size = size
        self.step = 0
        self.device = device

    def add(self, 
            state: np.ndarray, 
            actions: int, 
            action_log_probs: torch.Tensor, 
            next_state: np.ndarray, 
            rewards: int, 
            value_preds: torch.Tensor, 
            dones: bool
            ) -> None:
        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        value_pred = value_preds.squeeze()
        self.states[self.step].copy_(state)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.next_states[self.step].copy_(next_state)
        self.rewards[self.step].copy_(rewards)
        self.value_preds[self.step].copy_(value_pred)
        self.dones[self.step].copy_(dones)
        self.step = (self.step + 1) % self.size

    def sample(self, num_mini_batch: int) -> Tuple:
        assert self.size >= num_mini_batch
        
        mini_batch_size = self.size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(self.size)), mini_batch_size, drop_last=False)
        
        for indices in sampler:
            states = self.states.reshape(-1, *self.states.shape[1:])[indices]
            next_states = self.next_states.reshape(-1, *self.next_states.shape[1:])[indices]
            actions = self.actions.reshape(-1, self.actions.shape[-1])[indices]
            action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])[indices]
            advantages = self.advantages.reshape(-1)[indices]
            returns = self.returns.reshape(-1)[indices]
            values = self.value_preds.reshape(-1)[indices]
            
            yield states, next_states, actions, action_log_probs, returns, values, advantages

    def calculate_advantage(self, next_value: torch.Tensor, gamma: float) -> None:
        self.returns[-1] = next_value
        for step in reversed(range(self.size)):
            self.returns[step] = self.rewards[step] + gamma * self.returns[step + 1] * (1 - self.dones[step])
            self.advantages[step] = self.returns[step] - self.value_preds[step]

    def calculate_gae_advantage(self, next_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        self.returns[-1] = next_value
        for step in reversed(range(self.size)):
            self.returns[step] = self.rewards[step] + gamma * self.returns[step + 1] * (1 - self.dones[step])
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * (1 - self.dones[step]) - self.value_preds[step]
            self.advantages[step] = delta + gamma * gae_lambda * self.advantages[step + 1] * (1 - self.dones[step])
            self.returns[step] = self.advantages[step] + self.value_preds[step]
            # self.returns[step] = self.rewards[step] + gamma * self.returns[step + 1] * (1 - self.dones[step])