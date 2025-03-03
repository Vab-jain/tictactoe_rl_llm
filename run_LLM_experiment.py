from llm_eval import llm_play_eval

def main():
    experiments = [
        {
            'experiment_name': 'E14',
            'seed': None,
            'llm_model_id' : 'ollama_chat/llama3.1:latest',
            'load_llm_path' : 'saved_optim_llm/optim_llama3_1_8b__1D_int__metricV2__top1__dqn2.pkl',
        },
        # {
        #     'experiment_name': 'E1',
        #     'seed': None,
        #     'llm_model_id' : 'ollama_chat/llama3.2:3b',
        #     'load_llm_path' : 'saved_optim_llm/optim_llama3_1_8b__1D_int__metricV2__top1__dqn2.pkl',
        # },
        # {
        #     'experiment_name': 'E12',
        #     'seed': None,
        #     'llm_model_id' : 'ollama_chat/llama3.2:3b',
        #     'load_llm_path' : 'saved_optim_llm/optim_llama3_3b__1D_int__metricV2__top3__dqn2.pkl',
        # },
    ]
    
    for experiment_config in experiments:
        llm_play_eval(experiment_config)

if __name__ == '__main__':
    main()