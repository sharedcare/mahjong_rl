import numpy as np
from typing import Tuple
from pandas import Categorical
import torch
import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_space, n_actions, hidden_size=128, device=torch.device("cpu")):
        super(ActorCriticNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(np.array(state_space).prod(), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Linear(hidden_size, 1)

        self.to(device)

    def act(self, state) -> Tuple[float, torch.Tensor]:
        """Sample an action from state."""
        state_feature = self.layers(state)
        action_probs = self.actor(state_feature)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_probs = dist.log_prob(action_probs)

        return action.item(), action_log_probs

    def evaluate(self, state, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_feature = self.layers(state)
        value = self.critic(state_feature)

        action_probs = self.actor(state_feature)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return value, action_log_probs, dist_entropy

    def get_value(self, state) -> torch.Tensor:
        state_feature = self.layers(state)
        value = self.critic(state_feature)
        return value

    def forward(self, obs) -> torch.Tensor:
        state_feature = self.layers(obs)
        return state_feature


class ActorCriticCNNNetwork(nn.Module):
    def __init__(self, state_space, n_actions, hidden_size=512, device=torch.device("cpu")):
        super(ActorCriticCNNNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(state_space[-1], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_size),
            nn.ReLU()
        )

        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

        self.to(device)

    def act(self, state) -> Tuple[float, torch.Tensor]:
        """Sample an action from state."""
        # state_feature = self.layers(state)
        action_probs = self.actor(self.forward(state))
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_probs = dist.log_prob(action_probs)

        return action.item(), action_log_probs

    def evaluate(self, state, action) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # state_feature = self.layers(state)
        value = self.critic(self.forward(state))

        action_probs = self.actor(self.forward(state))
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return value, action_log_probs, dist_entropy

    def get_value(self, state) -> torch.Tensor:
        # state_feature = self.layers(state)
        value = self.critic(self.forward(state))
        return value

    def forward(self, obs) -> torch.Tensor:
        state_feature = self.layers(obs / 255.)
        return state_feature