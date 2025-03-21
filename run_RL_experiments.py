# experiments.py
from train_eval_handlers import train_agent, eval_agent
from utils import log_experiment

def main():
    experiments = [
        {
            #### global
            'experiment_name': 'E17-sig_v2__1D__LLM70b_GROQ_expert_opponent',
            'seed': 3,
            'env_policy' : 'dqn',
            #### llm
            'llm_model_id' : 'llama3-70b-8192',
            'use_GROQ' : True,
            'llm_cache_path' : 'LLM_database/E16-GROQ-llama3-70b.json',
            #### train
            'train_num_episodes': 1000,
            'train_llm_use_probability': 1,
            #### eval
            'eval_num_games': 100,
            'eval_llm_use_probability' : 1,
        },
        # {
        #     #### global
        #     'experiment_name': 'E16-sig_v2__1D__LLM70b_GROQ_seed3',
        #     'seed': 3,
        #     #### llm
        #     'llm_model_id' : 'llama3-70b-8192',
        #     'use_GROQ' : True,
        #     'llm_cache_path' : 'LLM_database/E16-GROQ-llama3-70b.json',
        #     #### train
        #     'train_num_episodes': 1000,
        #     'train_llm_use_probability': 1,
        #     #### eval
        #     'eval_num_games': 100,
        #     'eval_llm_use_probability' : 1,
        # },
        ############################################################
        # {
        #     #### global
        #     'experiment_name': 'E16-sig_v2__1D__LLM70b_GROQ_seed3_noEval',
        #     'seed': 3,
        #     #### llm
        #     'llm_model_id' : 'llama3-70b-8192',
        #     'use_GROQ' : True,
        #     'llm_cache_path' : 'LLM_database/E16-GROQ-llama3-70b.json',
        #     #### train
        #     'train_num_episodes': 0,
        #     'train_llm_use_probability': 0,
        #     #### eval
        #     'eval_num_games': 100,
        #     'eval_llm_use_probability' : 0,
        #     'eval_agent_load_name' : 'E16-sig_v2__1D__LLM70b_GROQ_seed3'
        # },
        ############################################################
        ]

    for experiment_config in experiments:
        train_info = train_agent(experiment_config)
        eval_info = eval_agent(experiment_config)
        log_experiment(experiment_config, train_info, eval_info)

if __name__ == '__main__':
    main()


