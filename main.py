import os
import argparse

import numpy as np
import torch
from tqdm import tqdm, trange

import rlcard
from rlcard.utils import (
    set_seed,
    tournament,
    reorganize,
    plot_curve,
)

from ppo_agent import PPOAgent
from utils import get_device, Logger

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

    # Set agents
    agent = PPOAgent(
        env=env, 
        state_shape=env.state_shape[0],
        num_actions=env.num_actions,
        device=device,
    )

    agents = [agent]
    for _ in range(1, env.num_players):
        agents.append(PPOAgent(
            env=env, 
            state_shape=env.state_shape[0],
            num_actions=env.num_actions,
            device=device,
        ))
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        t = trange(args.num_episodes)
        for episode in t:
            t.set_description("Episode %i" % episode)
            state, player_id = env.reset()
            episode_reward = 0
            done = False
            for i in range(1, env.num_players):
                agents[i].old_policy.load_state_dict(agents[0].old_policy.state_dict())
            while not done:
                if player_id == 0:
                    obs = state['obs']
                    legal_actions = list(state['legal_actions'].keys())
                    action, action_log_probs, value = agents[player_id].choose_action(obs, legal_actions)
                else:
                    action = agents[player_id].step(state)
                # Generate data from the environment
                next_state, next_player_id = env.step(action, agents[player_id].use_raw)
                payoffs = env.get_payoffs()
                reward = payoffs[player_id]
                done = env.is_over()
                if player_id == 0:
                    obs = state['obs']
                    next_obs = next_state['obs']
                    agents[player_id].memory.add(obs, action, action_log_probs, next_obs, reward, value, done)
                state = next_state
                player_id = next_player_id
                episode_reward += reward

            last_value = agents[0].get_value(agents[0].memory.next_states[-1].unsqueeze(0))
        
            action_loss, value_loss, entropy_loss, loss = agents[0].update(last_value)
            agents[0].memory.reset()
            
            t.set_postfix(loss=loss, episode_reward=episode_reward)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    env.timestep,
                    tournament(
                        env,
                        args.num_eval_games,
                    )[0],
                    loss
                )

            if episode != 0 and episode % args.save_every == 0:
                # Save model
                save_path = os.path.join(args.log_dir, 'ppo_model.pth')
                agents[0].save(save_path)

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

        # Plot the learning curve
        plot_curve(csv_path, fig_path, 'ppo')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("PPO agent for Mahjong")

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
        default='experiments/mahjong_ppo_sp_result/',
    )

    args = parser.parse_args()
    train(args=args)
    