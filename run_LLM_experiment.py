from llm_eval import llm_play_eval
from config import update_config

def main():
    experiments = [
        # {
        #     'experiment_name': 'E20-llama3.2:3b-ZS',
        #     'seed': 42,
        #     'llm_model_id' : 'ollama_chat/llama3.2:3b',
        #     # 'llm_model_id' : 'llama3-70b-8192',
        #     'eval_num_games' : 100,
        #     'use_GROQ' : False,
        #     ######## update config params
        #     'board_representation' : '1D',
        #     'cross_representation' : '1/-1',
        #     'task_description' : 'td_og',
        # },
        {
            'experiment_name': 'E20-llama2:13b-ZS',
            'seed': 42,
            'llm_model_id' : 'ollama_chat/llama2:13b',
            # 'llm_model_id' : 'llama3-70b-8192',
            'eval_num_games' : 100,
            'use_GROQ' : False,
            ######## update config params
            'board_representation' : '1D',
            'cross_representation' : '1/-1',
            'task_description' : 'td_og',
        },
        {
            'experiment_name': 'E20-llama3:70b-ZS',
            'seed': 42,
            # 'llm_model_id' : 'ollama_chat/llama3.2:3b',
            'llm_model_id' : 'llama3-70b-8192',
            'eval_num_games' : 100,
            'use_GROQ' : False,
            ######## update config params
            'board_representation' : '1D',
            'cross_representation' : '1/-1',
            'task_description' : 'td_og',
        },
        
    ]
    
    for experiment_config in experiments:
        update_config(experiment_config['board_representation'], experiment_config['cross_representation'], experiment_config['task_description'])
        llm_play_eval(experiment_config)

if __name__ == '__main__':
    main()