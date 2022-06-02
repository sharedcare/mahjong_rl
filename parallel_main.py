from multiprocessing import shared_memory
import os
import argparse
from git import WorkTreeRepositoryUnsupported

import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm, trange

import rlcard
from rlcard.utils import (
    set_seed,
    tournament,
    reorganize,
    plot_curve,
)

from dppo import DPPOWorker
from utils import get_device, Logger, Counter
from network import ActorCriticSeparateNetwork

def train_worker(args, env, id, shared_policy, shared_optim, counter, device):
    # set workers
    workers = []
    for _ in range(env.num_players):
        workers.append(DPPOWorker(
            state_shape=env.state_shape[0],
            num_actions=env.num_actions,
            device=device,
            shared_policy=shared_policy,
            shared_optim=shared_optim,
        ))
    env.set_agents(workers)

    while counter.get() < args.num_episodes:
        # t.set_description("Episode %i" % episode)
        state, player_id = env.reset()
        episode_reward = 0
        done = False
        for i in range(0, env.num_players):
            if i != id:
                workers[i].policy.load_state_dict(workers[id].policy.state_dict())
        while not done:
            if player_id == id:
                obs = state['obs']
                legal_actions = list(state['legal_actions'].keys())
                action, action_log_probs, value = workers[player_id].choose_action(obs, legal_actions)
            else:
                action = workers[player_id].step(state)
            # Generate data from the environment
            next_state, next_player_id = env.step(action, workers[player_id].use_raw)
            payoffs = env.get_payoffs()
            reward = payoffs[player_id]
            done = env.is_over()
            if player_id == id:
                obs = state['obs']
                next_obs = next_state['obs']
                workers[player_id].memory.add(obs, action, action_log_probs, next_obs, reward, value, done)
            state = next_state
            player_id = next_player_id
            episode_reward += reward

        last_value = workers[id].get_value(workers[id].memory.next_states[-1].unsqueeze(0))
    
        action_loss, value_loss, entropy_loss, loss = workers[id].update(last_value)
        workers[id].memory.reset()
        counter.reset()
        
        # t.set_postfix(loss=loss, episode_reward=episode_reward)


def train(args):
    # check if gpu is available
    device = torch.device("cpu")

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment
    env = rlcard.make(
        'mahjong',
        config={
            'seed': args.seed,
        }
    )

    shared_policy = ActorCriticSeparateNetwork(env.state_shape[0], env.num_actions, 64, device)
    shared_optimizer = optim.Adam(shared_policy.parameters(), lr=0.001)

    counter = Counter()
    processes = []
    # Run parallel agents training
    for id in range(env.num_players):
        p = mp.Process(target=train_worker, args=(args, env, id, shared_policy, shared_optimizer, counter, device))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Distributed PPO agent for Mahjong")

    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10000,
    )

    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )

    parser.add_argument(
        '--save_every',
        type=int,
        default=1000,
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/mahjong_dppo_sp_result/',
    )

    args = parser.parse_args()
    train(args=args)