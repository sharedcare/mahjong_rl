''' A toy example of playing against rule-based bot on UNO
'''
import os
import torch
import numpy as np
import rlcard
from rlcard import models
from wandb import agent
from human_agent import HumanAgent, _print_action
from ppo_agent import PPOAgent

# Make environment
env = rlcard.make('mahjong')
human_agent = HumanAgent(env.num_actions)
device = torch.device("cpu")
ppo_agent = PPOAgent(
        env=env, 
        state_shape=env.state_shape[0],
        num_actions=env.num_actions,
        device=device,
    )

model_path = os.path.join('experiments/mahjong_ppo_sp_result/', 'ppo_model.pth')
ppo_agent.restore(model_path)
agents = [human_agent]
for _ in range(1, env.num_players):
        agents.append(ppo_agent)
env.set_agents(agents)

print(">> Mahjong ")

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state['player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses ', end='')
        _print_action(pair[1])
        print('')

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win!')
    else:
        print('You lose!')
    
    if payoffs[0] == 0:
        print('No player win the game')
    else:
        print('Player {} win the game!'.format(np.argmax(payoffs) + 1))
    print('')
    input("Press any key to continue...")