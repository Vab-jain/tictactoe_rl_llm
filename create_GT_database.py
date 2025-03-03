import numpy as np
import random
import time
import os

from env import TicTacToeEnv
from agent import DQNAgent



def get_board_representation(board):
    # Generate a prompt for the LLM based on the board state
    symbols = {1: '1', -1: '-1', 0: '0'}
    # symbols = {1: 'X', -1: 'O', 0: '_'}
    available_actions = ''

    """
    1d board representation 
    [0,0,0,0,1,0,0,0,0]
    
    No available actions
    """
    # board_str = '' 
    # for i in range(3): 
    #     row = ' '.join([str(symbols[board[i, j]]) for j in range(3)]) 
    #     board_str += row + ' '
    # available_actions = None
    
    """
    1d board representation 
    [0,0,0,0,1,0,0,0,0]
    
    with Available actions
    """
    board_str = '' 
    for i in range(3): 
        row = ' '.join([str(symbols[board[i, j]]) for j in range(3)]) 
        board_str += row + ' '
        for j in range(3):
            if board[i,j]==0:
                available_actions += f" {i*3+j}"
    
    """
    2d board representation 
    [[0 0 0], [0 0 0], [0 0 0], ]
    
    with Available actions
    """
    # board_str = '' 
    # for i in range(3): 
    #     row = ' '.join([str(symbols[board[i, j]]) for j in range(3)]) 
    #     board_str += '[' + row + '], '
    #     for j in range(3):
    #         if board[i,j]==0:
    #             available_actions += f" {i*3+j}"
    
    return board_str, available_actions


experiment_name = 'EA-DQN2'
num_games = 100
agent_plays = 'random' 

# Create directories to load results
results_dir = os.path.join('experiments_results/RL_phase', experiment_name)

# craete environment
env = TicTacToeEnv()

# load the agent
agent_filepath = os.path.join(results_dir, 'best_tic_tac_toe_agent.pth')
agent = DQNAgent(
        state_size=env.observation_space.shape[0], 
        action_size=env.action_space.n.item(), 
    )
agent.load(filepath=agent_filepath)
agent.eval()

win, loss, draw = 0, 0, 0

with open("train_data.txt", "a") as f:
    for _ in range(num_games):
        # Determine who plays first based on agent_plays
        if agent_plays == 'first_player':
            first_player = 1  # RL agent plays first
        elif agent_plays == 'second_player':
            first_player = -1  # Envoronment agent plays first
        else:  # Randomly assign who plays first
            first_player = random.choice([1, -1])

        state = env.reset(first_player=first_player)
        done = False
        while not done:
            '''
            For single action
            '''
            # action = agent.select_action(state)  # No exploration during evaluation
            # next_state, _, done = env.step(action)
            
            # # log state-text and action-text here 
            # ###
            # board_str, _ = get_board_representation(state.reshape(3, 3))
            # f.write(f"{board_str}\t{action}\n")
            '''
            For Top-3 action
            '''
            top_actions = agent.select_top3_action(state)  # Get top 3 actions
            action = top_actions[0]  # Pick the top action to play
            next_state, _, done = env.step(action)
            
            # log state-text and action-text here 
            ###
            board_str, _ = get_board_representation(state.reshape(3, 3))
            f.write(f"{board_str}\t{top_actions}\n")
            '''
            For q_values
            '''
            # action, q_values = agent.get_q_values(state)  # No exploration during evaluation
            # next_state, _, done = env.step(action)
            
            # # log state-text and q_values-text here 
            # ###
            # board_str, _ = get_board_representation(state.reshape(3, 3))
            # f.write(f"{board_str}\t{q_values}\n")
            
            # change state
            state = next_state

        
        # Update stats
        if env.get_winner() == 1:
            win += 1
        elif env.get_winner() == -1:
            loss +=1
        elif env.get_winner() == 0:
            draw +=1

print(f"Win: {win}, Loss: {loss}, Draw: {draw}")

