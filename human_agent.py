from tabnanny import check
from typing import List
from termcolor import colored
from rlcard.games.mahjong.card import MahjongCard
from rlcard.games.mahjong.utils import *
from pprint import pprint

class HumanAgent(object):
    ''' A human agent for Leduc Holdem. It can be used to play against trained models
    '''

    def __init__(self, num_actions):
        ''' Initilize the human agent

        Args:
            num_actions (int): the size of the ouput action space
        '''
        self.use_raw = True
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        ''' Human agent will display the state and make decisions through interfaces

        Args:
            state (dict): A dictionary that represents the current state

        Returns:
            action (str): The action decided by human
        '''
        # print(state['raw_obs'])
        valid_act = state['raw_obs']['valid_act']
        action = valid_act[0]
        if len(valid_act) > 1:
            print('=============== Last Card ===============')
            print_mahjong_cards([state['raw_obs']['table'][-1]])
            print('')
            print('-------- Choose An Action --------')
            for i, act in enumerate(valid_act):
                print('{}: {}'.format(i, act), end='')
                if i < len(valid_act) - 1:
                    print(', ', end='')
            print('\n')
            action_idx = int(input('>> Choose an action (integer): '))
            while action_idx < 0 and action_idx >= len(valid_act):
                print('Action illegel...')
                action_idx = int(input('>> Re-choose action (integer): '))

            action = valid_act[action_idx]

        if action == 'play':
            _print_state(state['raw_obs'], state['action_record'], state['raw_legal_actions'])
            card = int(input('>> You choose card to play (integer): '))
            
            while not _check_action(state, card):
                print('Action illegel...')
                card = int(input('>> Re-choose card (integer): '))
            
            print_mahjong_cards([state['raw_legal_actions'][card]])
            print('')
            return state['raw_legal_actions'][card]

        else:
            return action

    def eval_step(self, state):
        ''' Predict the action given the curent state for evaluation. The same to step here.

        Args:
            state (numpy.array): an numpy array that represents the current state

        Returns:
            action (str): the action predicted (randomly chosen) by the random agent
        '''
        return self.step(state), {}


def print_mahjong_cards(cards: List[MahjongCard], is_colored=True):
    for card in cards:
        type = card.type
        trait = card.trait
        color = None
        if type == 'dots':
            color = 'blue'
        elif type == 'bamboo':
            color = 'green'
        elif type == 'characters':
            color = 'red'

        if is_colored and color:
            print(colored('[{}-{}]'.format(type, trait), color), end='  ')
        else:
            print('[{}-{}]'.format(type, trait), end='  ')

def _print_state(state, action_record, legal_actions):
    ''' Print out the state of a given player

    Args:
        player (int): Player id
    '''
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == state['player']:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses ', end='')
        _print_action([pair[1]])
        print('')

    print('\n=============== Your Hand ===============')
    print_mahjong_cards(state['current_hand'])
    print('')
    print('=============== Table ===============')
    print_mahjong_cards(state['table'])
    print('')
    print('=============== Player Pile ===============')
    piles = state['players_pile']
    for player_id, pile in piles.items():
        print('--- Player {} ---'.format(player_id + 1))
        for cards in pile:
            print_mahjong_cards(cards)
        print('')
    print('')
    print('======== Actions You Can Choose =========')
    for i, action in enumerate(legal_actions):
        print(str(i)+': ', end='')
        print_mahjong_cards([action])
        if i < len(legal_actions) - 1:
            print(', ', end='')
    print('\n')

def _print_action(action):
    ''' Print out an action in a nice form

    Args:
        action (list): A string a action
    '''
    if isinstance(action, MahjongCard):
        print_mahjong_cards([action])
    elif isinstance(action[0], str):
        print(action[0])
    else:
        print_mahjong_cards(action)

def _check_action(state, action_idx):
    legal_actions = state['legal_actions']
    raw_legal_actions = state['raw_legal_actions']
    str_legal_actions = cards2list(raw_legal_actions)
    action_str = str_legal_actions[action_idx]
    action = card_encoding_dict[action_str]
    return action in legal_actions
