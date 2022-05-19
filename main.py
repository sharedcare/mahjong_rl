import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

from ppo_agent import PPOAgent

def train(args):
    # check if gpu is available
    device = get_device()

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
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            state, player_id = env.reset()
            episode_reward = 0
            done = False

            while not done:
                if player_id == 0:
                    obs = state['obs']
                    action, action_log_probs, value = agents[player_id].choose_action(obs)
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

            with torch.no_grad():
                next_value = agents[0].policy.get_value(agents[0].memory.states[-1])
        
            samples = agents[0].memory.sample(agents[0].num_mini_batch)
            action_loss, value_loss, entropy_loss, loss = agents[0].update(next_value, samples)
            
            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    env.timestep,
                    tournament(
                        env,
                        args.num_eval_games,
                    )[0]
                )

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
        default=5000,
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
        '--log_dir',
        type=str,
        default='experiments/mahjong_ppo_result/',
    )

    args = parser.parse_args()
    train(args=args)
    