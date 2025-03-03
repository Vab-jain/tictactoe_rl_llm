import dspy
from dspy.evaluate import Evaluate
import random

import config
import utils
from utils import GenerateAction, TaskContext



def prompt_tune_llm(
                llm_model_id,
                GT_dataset,
                llm_save_path,
                board_representation=config.board_representation,
                cross_representataion=config.cross_representation,
                ):
    '''
    Function to optimize LLM
    
    Args:
        llm_model_id  : 
        GT_dataset  :
        llm_save_path :
        board_representation : (default) '1D',
        cross_representataion : (default) '1/-1'
    
    Return:
        llm_accuracy : 
        optim_llm_accuracy : 
        (saves optimied LLM at 'llm_save_path')
    '''
    
    # configure LLM
    lm = dspy.LM(llm_model_id, api_base='http://localhost:11434',cache=False)
    dspy.configure(lm=lm)
    llm_agent = dspy.ChainOfThought(GenerateAction)

    print('LLM Configured!')

    # Create the Task-Context
    taskcontext = TaskContext(board_representation,cross_representataion,)

    # load the dataset
    df = utils.read_txt_to_dataframe(GT_dataset)
    trainset, devset = [], []
    train_size = int(0.2 * len(df))  # First 20% for trainset
    for i, row in df.iterrows():
        # Create the example using the board_str and action
        example = dspy.Example(context=taskcontext.context, current_state=row["board_str"], answer=row["actions"]).with_inputs('context','current_state')
        if i<train_size:
            trainset.append(example)
        else:
            devset.append(example)
    # shuffle trainset and testset
    random.shuffle(trainset)
    random.shuffle(devset)

    print('GT Dataset Loaded!')
    print(f'Train Set : {len(trainset)}  |  Validation Set : {len(devset)}')

    # Set up the evaluator, which can be re-used in your code.
    evaluator = Evaluate(devset=devset, num_threads=1, display_progress=True, display_table=5)

    print(' Evaluating Original LLM (before optimization)!')
    # Launch evaluation for the original LLM
    llm_accuracy = evaluator(llm_agent, metric=utils.validate_answer)

    # LLM Optimization / Prompt Tuning
    tp = dspy.MIPROv2(metric=utils.validate_answer, auto="light", num_threads=24, )
    optimized_llm_agent = tp.compile(llm_agent, trainset=trainset, requires_permission_to_run=False)

    print(' Evaluating Optimized LLM (after optimization)!')
    # Launch evaluation for the optimized LLM
    optim_llm_accuracy = evaluator(optimized_llm_agent, metric=utils.validate_answer)

    # save optim_LLM
    optimized_llm_agent.save(llm_save_path)
    print('LLM saved successfully at: ', llm_save_path)

    return llm_accuracy, optim_llm_accuracy


if __name__=='__main__':
    '''
    Checklist for this experiment:
        - config.py 
            - Board Representation: utils.TaskContext
            - Cross Representation: utils.TaskContext
            - Generate Action : utils.GenerateAction
        
    '''

    # configuration
    llm_model_id = 'ollama_chat/llama3.2:3b'
    GT_dataset = "GT_database/GT_dqn2_1D_int_q_values.txt"
    root_path = 'saved_optim_llm/optim_llama3_3b__1D_int__metricV3__q_values__dqn2'
    llm_save_path = f'{root_path}.pkl'
    result_path = f'{root_path}.txt'
    
    llm_accuracy, optim_llm_accuracy = prompt_tune_llm(llm_model_id, GT_dataset, llm_save_path)
    
    with open(result_path, 'w') as f:
        f.write('Experiment Details:\n')
        f.write('-------------------\n')
        f.write(f'llm_model_id : {llm_model_id}\n')
        f.write(f'GT_dataset : {GT_dataset}\n')
        f.write(f'llm_save_path : {llm_save_path}\n\n')
        f.write(f'LLM Accuracy (before optimization): {llm_accuracy}\n')
        f.write(f'Optimized LLM Accuracy (after optimization): {optim_llm_accuracy}\n')
        