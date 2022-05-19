import numpy as np
from typing import Tuple
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class Memory(object):
    def __init__(self, size, state_shape, action_space, device=torch.device("cpu")) -> None:
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

    def add(self, state, actions, action_log_probs, next_state, rewards, value_preds, dones) -> None:
        self.states[self.step].copy_(state)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.next_state[self.step].copy_(next_state)
        self.rewards[self.step].copy_(rewards)
        self.value_preds[self.step] = value_preds.copy()
        self.dones[self.step].copy_(dones)
        self.step = (self.step + 1) % self.size

    def sample(self, num_mini_batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_steps, n_threads = self.rewards.shape[0:2]
        batch_size = n_steps * n_threads
        
        assert batch_size >= num_mini_batch
        
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        
        for indices in sampler:
            states = self.states.reshape(-1, *self.states.shape[-1])[indices]
            next_states = self.next_states.reshape(-1, self.next_states.shape[-1])[indices]
            actions = self.actions.reshape(-1, self.actions.shape[-1])[indices]
            action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])[indices]
            advantages = self.advantages.reshape(-1)[indices]
            returns = self.returns.reshape(-1)[indices]
            values = self.value_preds.reshape(-1)[indices]
            
            yield states, next_states, actions, action_log_probs, returns, values, advantages

    def calculate_advantage(self, next_value, gamma) -> None:
        self.returns[-1] = next_value
        for step in reversed(range(self.size)):
            self.returns[step] = self.rewards[step] + gamma * self.returns[step + 1] * (1 - self.dones[step])
            self.advantages[step] = self.returns[step] - self.value_preds[step]

    def calculate_gae_advantage(self, next_value, gamma, gae_lambda) -> None:
        self.value_preds[-1] = next_value
        for step in reversed(range(self.size)):
            delta = self.rewards[step] + gamma * self.value_preds[step + 1] * (1 - self.dones[step]) - self.value_preds[step]
            self.advantages[step] = delta + gamma * gae_lambda * self.advantages[step + 1] * (1 - self.dones[step])
            self.returns[step] = self.advantages[step] + self.value_preds[step]
            # self.returns[step] = self.rewards[step] + gamma * self.returns[step + 1] * (1 - self.dones[step])