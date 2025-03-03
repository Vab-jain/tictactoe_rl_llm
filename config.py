# config.py
'''
Global configurations for all experiments
'''

# accessed in: prompt_tune.py, wrappers.py
board_representation = '1D'      
# board_representation = '2D'

# accessed in: prompt_tune.py, wrappers.py
cross_representation = '1/-1'
# cross_representation = 'X/O'

# defined in llm_config.py
# accessed in: prompt_tune.py, wrappers.py
dspy_signature = 'v1'

# defined in dspy_metric.py
# accessed in: prompt_tune.py
validation_metric = 'v3'