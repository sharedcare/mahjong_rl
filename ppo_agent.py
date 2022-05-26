import argparse
import pprint
from sre_parse import State
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import namedtuple
from copy import deepcopy

import rlcard
from rlcard.utils.utils import remove_illegal
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

from memory import Memory
from network import ActorCriticNetwork, ActorCriticCNNNetwork

class PPOAgent(object):
    def __init__(self, 
                 env, 
                 state_shape, 
                 num_actions, 
                 device, 
                 gamma=0.99, 
                 alpha=0.001, 
                 gae_lambda=0.95, 
                 clip_factor=0.2, 
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 num_mini_batch=16,
                 mem_size=512,
                 hidden_size=64, 
                 K_epochs=1000) -> None:
        self.gamma = gamma
        self.lr = alpha
        self.gae_lambda = gae_lambda
        self.clip_factor = clip_factor
        self.K_epochs = K_epochs
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.device = device

        self.env = env

        # state_space = env.observation_space.shape
        # n_actions = env.action_space.n

        self.memory = Memory(mem_size, state_shape, num_actions)

        self.policy = ActorCriticNetwork(state_shape, num_actions, hidden_size, device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.old_policy = ActorCriticNetwork(state_shape, num_actions, hidden_size, device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()

        # rlcard arguments
        self.use_raw = False
        self.num_actions = num_actions

    def restore(self, model_path: str) -> None:
        ''' Load model from given path. 

        Args:
            model_path : Path to model weights
        '''
        print('Load model from', model_path)
        pretained_model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.policy.load_state_dict(pretained_model)
        self.old_policy.load_state_dict(self.policy.state_dict())

    def save(self, save_path: str) -> None:
        ''' Save model in designated path. 

        Args:
            save_path : Path to save model
        '''
        torch.save(self.policy.state_dict(), save_path)
        # print('Model saved in', save_path)

    def step(self, state: Dict) -> int:
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state : A dictionary that represents the current state
        Returns:
            action : The action predicted by the ppo agent
        '''
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        action, _, _ = self.choose_action(obs, legal_actions)
        return int(action)

    def eval_step(self, state: Dict) -> Tuple[int, Dict]:
        ''' Predict the action given the current state for evaluation.

        Args:
            state : A dictionary that represents the current state
        Returns:
            action : The action predicted by the ppo agent
            probs : The list of action probabilities
        '''
        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1/len(state['legal_actions'])

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info

    def get_state(self, obs: np.ndarray) -> torch.Tensor:
        ''' Formulate and convert numpy state to torch tensor.

        Args:
            obs : A numpy array that represents the current state
        Returns:
            state : A torch tensor of the reformed state
        '''
        state = obs.transpose((2, 0, 1))
        state = torch.from_numpy(state).float().to(self.device)
        return state.unsqueeze(0)

    def choose_action(self, state: np.ndarray, legal_actions: List) -> Tuple[float, torch.Tensor, torch.Tensor]:
        ''' Choose action from the given state using old policy.

        Args:
            state : A numpy array that represents the current state
            legal_actions : A list of available actions
        Returns:
            action : Choosed action from actor
            action_log_probs : Log of action probabilities
            value : Critic value
        '''
        self.old_policy.eval()
        with torch.no_grad():
            # state = self.get_state(state)
            state = torch.from_numpy(state).float().to(self.device)
            state = state.unsqueeze(0)
            action, action_log_probs = self.old_policy.act(state, legal_actions)
            value = self.old_policy.get_value(state)

        return action, action_log_probs, value

    def get_value(self, state: torch.Tensor):
        ''' Get critic value from the given state using current policy.

        Args:
            state : A torch tensor that represents the current state
        Returns:
            value : Critic value
        '''
        self.policy.eval()
        with torch.no_grad():
            value = self.policy.get_value(state)

        return value

    def update(self, next_value: torch.Tensor, samples: Tuple) -> Tuple[List[float], List[float], List[float], List[float]]:
        ''' PPO update.

        Args:
            next_value : Value of the last state in the memory buffer
            samples: Samples from memory
        Returns:
            action_loss : Action loss
            value_loss : Value loss
            entropy_loss : Entropy loss
            loss: Mean value of total loss
        '''
        action_losses = []
        value_losses = []
        entropy_losses = []
        losses = []
        # self.memory.calculate_gae_advantage(next_value, self.gamma, self.gae_lambda)
        self.memory.calculate_advantage(next_value, self.gamma)
        self.policy.train()
        
        for _ in range(self.K_epochs):
            for states, next_states, actions, old_action_log_probs, returns, values, advantages in samples:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                new_values, action_log_probs, dist_entropy = self.policy.evaluate(states, actions)

                # calculate policy loss
                ratio = torch.exp(action_log_probs - old_action_log_probs)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_factor, 1.0 + self.clip_factor) * advantages
                
                action_loss = -1 * torch.min(surr1, surr2).mean()

                # calculate value loss
                value_loss_unclipped = torch.square(new_values - returns)
                values_clipped = values + torch.clamp(new_values - values, -self.clip_factor, self.clip_factor)
                value_loss_clipped = torch.square(values_clipped - returns)
                value_loss = self.value_loss_coef * torch.mean(torch.max(value_loss_clipped, value_loss_unclipped))
                # value_loss = self.value_loss_coef * self.MSE_loss(returns, new_values.squeeze())

                entropy_loss = self.entropy_coef * dist_entropy.mean()

                # total loss
                loss = action_loss + value_loss - entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                action_losses.append(action_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                losses.append(loss.item())

            # hard update
            self.old_policy.load_state_dict(self.policy.state_dict())

        return np.mean(action_losses), np.mean(value_losses), np.mean(entropy_losses), np.mean(losses)

    def train_atari(self, num_episodes: int) -> None:
        episode_rewards = []
        for e in tqdm(range(1, num_episodes + 1)):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, action_log_probs, value = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.memory.add(state, action, action_log_probs, next_state, reward, value, done)
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)

            with torch.no_grad():
                next_value = self.policy.get_value(self.memory.states[-1])
        
            samples = self.memory.sample(self.num_mini_batch)
            action_loss, value_loss, entropy_loss, loss = self.update(next_value, samples)

            if e != 0 and e % 100 == 0:
                print("Episode {}: score = {}, average score = {}, loss = {}".format(e, episode_reward, np.mean(episode_rewards[-10:]), loss))

    def train_rlcard(self, num_episodes: int) -> None:
        episode_rewards = []
        for e in tqdm(range(1, num_episodes + 1)):
            trajectories = [[] for _ in range(self.num_players)]
            state, player_id = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, action_log_probs, value = self.choose_action(state)
                # Generate data from the environment
                next_state, next_player_id = self.env.step(action, self.use_raw)
                payoffs = self.env.get_payoffs()
                reward = payoffs[player_id]
                done = self.env.is_over()
                if player_id == 0:
                    self.memory.add(state, action, action_log_probs, next_state, reward, value, done)
                state = next_state
                player_id = next_player_id
                episode_reward += reward
            
            episode_rewards.append(episode_reward)

            with torch.no_grad():
                next_value = self.policy.get_value(self.memory.states[-1])
        
            samples = self.memory.sample(self.num_mini_batch)
            action_loss, value_loss, entropy_loss, loss = self.update(next_value, samples)

            if e != 0 and e % 100 == 0:
                print("Episode {}: score = {}, average score = {}, loss = {}".format(e, episode_reward, np.mean(episode_rewards[-10:]), loss))


if __name__ == "__main__":
    # Make environment
    env = rlcard.make(
        "mahjong",
        config={
            'seed': 42,
        }
    )

    # Seed numpy, torch, random
    set_seed(42)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set agents
    agent = PPOAgent(env, env.state_shape[0], env.num_actions, device)
    # env.set_agents([agent for _ in range(env.num_players)])
    train_episodes = 10000
    agent.train_rlcard(train_episodes)

