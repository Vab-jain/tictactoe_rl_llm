# experiments.py
from train_eval_handlers import train_agent, eval_agent
from utils import log_experiment

def main():
    experiments = [
        {
            #### global
            'experiment_name': 'EA-DQN2.1',
            'seed': None,
            #### llm
            'llm_model_id' : 'ollama_chat/llama3.2:3b',
            #### train
            'train_num_episodes': 3000,
            'train_llm_use_probability': 0,
            #### eval
            'eval_num_games': 1000,
        },
        ############################################################
        # DEFAULT EXPERIMENT CONFIG
        # {
        #     'experiment_name': 'Experiment-X',
        #     'num_episodes': 1000,
        #     'llm_use_probability': 0,
        #     'agent_plays' : 'random',  # agent plays {first_player, second_player, random}
        #     'eval_num_games': 1000,
        #     'agent_policy': 'random',
        
        #     'seed': None,                 # default None
        #     'use_llm_cache' : False,      # default False
        #     'save_llm_cache' : False,     # default False
        #     'use_oracle' : False,         # default False
        #     'batch_size' : 32,            # default 32
        #     'learning_rate': 1e-4,        # default 1e-4
        #     'gamma' : 0.99,               # default 0.99
        #     'target_update' : 10,         # default 10
        #     'eval_llm_use_probability': 0.0, # default 0
        # },
    ]

    for experiment_config in experiments:
        train_info = train_agent(experiment_config)
        eval_info = eval_agent(experiment_config)
        log_experiment(experiment_config, train_info, eval_info)

if __name__ == '__main__':
    main()


