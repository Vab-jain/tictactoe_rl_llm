from gymnasium.core import Wrapper
from gymnasium import spaces
import numpy as np
import dspy
from utils import GenerateAction
import config
import json
import time


def configure_llm(llm_model_id='ollama_chat/llama3.2:3b', cache=False, GROQ=False):
    if GROQ:
        lm = dspy.LM(llm_model_id, api_base='https://api.groq.com/openai/v1', api_key=config.GROQ_API_KEY)
    else:
        lm = dspy.LM(llm_model_id, api_base='http://localhost:11434', cache=cache)
    dspy.configure(lm=lm)

# lm = dspy.LM('ollama_chat/llama3.2:3b', api_base='http://localhost:11434')
# dspy.configure(lm=lm)

class LLMSuggestionWrapper(Wrapper):
    def __init__(self, env, 
                 llm_use_probability=0, 
                 llm_model_id='ollama_chat/llama3.2:3b', 
                 load_llm_path=None, 
                 cache_llm = False, 
                 GROQ=False, 
                 llm_cache_path=None):
        super().__init__(env)
        
        # extend observation space        
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.int32)

        # setup LLM
        configure_llm(llm_model_id, cache_llm, GROQ)
        self.llm_agent = dspy.Predict(GenerateAction)
        # self.llm_agent = dspy.ChainOfThought(GenerateAction)
        
        if load_llm_path:
            self.llm_agent.load(path=load_llm_path)
        
        self.GROQ = GROQ
        self.llm_cache_dict = {}
        self.use_llm_cache = True if llm_cache_path is not None else False
        self.llm_cache_path = llm_cache_path
        if self.llm_cache_path:
            with open(llm_cache_path, 'r') as f:
                self.llm_cache_dict = json.load(f)
            
        self.context_sample = config.task_decription
        
        self.llm_use_prob = llm_use_probability

    
    def step(self, action):
        obs, reward, done = self.env.step(action)
        # add LLM suggstion to the observation
        llm_suggestion = -1     # LLM suggestion is -1 if no suggestion is made
        if self.llm_use_prob > np.random.random():
            llm_suggestion = self._get_llm_suggestion(obs)
        obs = np.concatenate((obs, [llm_suggestion]))    
        return obs, reward, done
    
    def reset(self, first_player=1):
        obs = self.env.reset(first_player)
        # add LLM suggstion to the observation
        llm_suggestion = -1     # LLM suggestion is -1 if no suggestion is made
        if self.llm_use_prob > np.random.random():
            llm_suggestion = self._get_llm_suggestion(obs)
        obs = np.concatenate((obs, [llm_suggestion])) 
        return obs
    
    def get_winner(self):
        return self.env.get_winner()
    
    def _get_llm_suggestion(self, state):
        # first check if the suggestion is available in the LLM cache
        if self.use_llm_cache:
            action = self.llm_cache_dict.get(json.dumps(state.tolist()), None)
            if action:
                return action
        # if llm suggestion not found in LLM cache, call LLM
        board_str, available_actions = self._get_board_representation(state.reshape(3, 3))
        if config.dspy_signature=='v1':
            response = self.llm_agent(context=self.context_sample, current_state=board_str)
        elif config.dspy_signature=='v2':
            response = self.llm_agent(context=self.context_sample, current_state=board_str, available_actions=available_actions)
        elif config.dspy_signature=='v3':
            response = self.llm_agent(current_state=board_str, available_actions=available_actions)
        if self.GROQ:
            time.sleep(5)
        action = int(response.answer)
        if self.use_llm_cache:
            self._save_state_action_pair(state.tolist(), action)
        return action
    
    def _save_state_action_pair(self, state, action):
        state_key = json.dumps(state)
        self.llm_cache_dict[state_key] = int(action)  # Convert state to a tuple for hashability
        with open(self.llm_cache_path, 'w') as f:
            json.dump(self.llm_cache_dict, f, indent=4)
    
    def _get_board_representation(self, board):
        # Generate a prompt for the LLM based on the board state
        if config.cross_representation=='1/-1':
            symbols = {1: '1', -1: '-1', 0: '0'}
        elif config.cross_representation=='X/O':
            symbols = {1: 'X', -1: 'O', 0: '_'}
        available_actions = ''

        """
        1d board representation 
        [0,0,0,0,1,0,0,0,0]
        
        with Available actions
        """
        if config.board_representation=='1D':
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
        if config.board_representation=='2D':
            board_str = '' 
            for i in range(3): 
                row = ' '.join([str(symbols[board[i, j]]) for j in range(3)]) 
                board_str += '[' + row + '], '
                for j in range(3):
                    if board[i,j]==0:
                        available_actions += f" {i*3+j}"
        
        return board_str, available_actions
    
    
    
class RandomSuggestionWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # extend observation space        
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.int32)

    
    def step(self, action):
        obs, reward, done = self.env.step(action)
        # add Random suggstion to the observation
        suggestion = self._get_random_suggestion()
        obs = np.concatenate((obs, [suggestion]))    
        return obs, reward, done
    
    def reset(self, first_player=1):
        obs = self.env.reset(first_player)
        # add Random suggstion to the observation
        suggestion = self._get_random_suggestion()
        obs = np.concatenate((obs, [suggestion]))  
        return obs
    
    def get_winner(self):
        return self.env.get_winner()

    def _get_random_suggestion(self):
        # get a random suggestion from the action space
        return np.random.choice(self.action_space.n)