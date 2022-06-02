'''
Distributed Proximal Policy Optimization(PPO)

https://arxiv.org/pdf/1707.02286.pdf

'''
from typing import Dict, List, Tuple
from imageio import save
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from memory import Memory
from network import ActorNetwork, CriticNetwork, ActorCriticSeparateNetwork
from utils import Counter, ensure_shared_grads


class DPPOWorker(object):
    def __init__(self,
                 state_shape, 
                 num_actions, 
                 device,
                 shared_policy,
                 shared_optim, 
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

        self.memory = Memory(mem_size, state_shape, num_actions)

        self.shared_policy = shared_policy
        self.policy = ActorCriticSeparateNetwork(state_shape, num_actions, hidden_size, device)
        self.optimizer = shared_optim
        
         # rlcard arguments
        self.use_raw = False
        self.num_actions = num_actions

    def _copy_shared_params(self):
        self.policy.load_state_dict(self.shared_policy.state_dict())

    def restore(self, model_path: str) -> None:
        ''' Load model from given path. 

        Args:
            model_path : Path to model weights
        '''
        print('Load model from', model_path)
        pretained_model = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.policy.load_state_dict(pretained_model)

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
        self.policy.eval()
        with torch.no_grad():
            # state = self.get_state(state)
            state = torch.from_numpy(state).float().to(self.device)
            state = state.unsqueeze(0)
            action, action_log_probs = self.policy.act(state, legal_actions)

        return action, action_log_probs

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

    def update(self, next_value: torch.Tensor, counter: Counter) -> Tuple[List[float], List[float], List[float], List[float]]:
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
        self.memory.calculate_gae_advantage(next_value, self.gamma, self.gae_lambda)
        self.policy.train()
        
        for _ in range(self.K_epochs):
            self._copy_shared_params()
            samples = self.memory.sample(self.num_mini_batch)
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
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                ensure_shared_grads(self.policy, self.shared_policy)
                self.optimizer.step()
                action_losses.append(action_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                losses.append(loss.item())
                counter.increment()

        return np.mean(action_losses), np.mean(value_losses), np.mean(entropy_losses), np.mean(losses)
