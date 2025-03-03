import os
import tqdm
import numpy as np

from env import TicTacToeEnv, LLMSuggestionWrapper

def llm_play_eval(
            experiment_config, 
            ):
    # Unpack experiment configuration
    experiment_name = experiment_config['experiment_name']
    llm_model_id = experiment_config['llm_model_id']
    load_llm_path = experiment_config['load_llm_path']
    eval_num_games = experiment_config.get('eval_num_games', 100)
    
    env = TicTacToeEnv()
    env = LLMSuggestionWrapper(env, 
                            llm_use_probability=1, 
                            llm_model_id=llm_model_id,
                            load_llm_path=load_llm_path,
                            )

    '''
        * Test the LLM suggestion accuracy *
        - Let LLM play the game
        - Count the number of wins, losses, draws, and invalid actions
    '''
    win_ctr = 0
    loss_ctr = 0
    draw_ctr = 0
    invalid_ctr = 0
    suggestion_ctr = 0

    for episode in tqdm.tqdm(range(eval_num_games)):
        first_player = np.random.choice([1,-1])
        state = env.reset(first_player)
        done = False
        while not done:
            suggestion_ctr += 1
            action = state[-1]
            if action==-1:
                reward = -10
                break
            state, reward, done = env.step(action)
        if reward==1:
            win_ctr += 1
        elif reward==-1:
            loss_ctr += 1
        elif reward==-10:
            invalid_ctr += 1
        elif reward==0:
            draw_ctr += 1
        else:
            exit('Unknown reward')

    print('Saving Results...')
    RESULTS_PATH = os.path.join("experiments_results", "LLM_phase", f"{experiment_name}.txt")
    with open(RESULTS_PATH, 'a') as file:
        file.write('Experiment Details:\n')
        file.write('-------------------\n')
        file.write(f'Experiment Name: {experiment_name}\n')
        file.write(f'LLM: {llm_model_id}\n')
        file.write('LLM Agent Plays: Random\n')
        file.write(f'Number of Games: {eval_num_games}\n')
        file.write('-------------------\n\n')

        file.write('LLM Agent Accuracy:\n')
        file.write(f'Win: {win_ctr}, Loss: {loss_ctr}, Draw: {draw_ctr}, Invalid: {invalid_ctr}, Invalid (%): {100*invalid_ctr/suggestion_ctr}\n')
        file.write('#####################\n\n')
    print(f'Results saved successfully at: {RESULTS_PATH}')