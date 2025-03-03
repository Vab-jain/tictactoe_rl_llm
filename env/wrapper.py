from gymnasium.core import Wrapper
from gymnasium import spaces
import numpy as np
import dspy
from utils import GenerateAction


def configure_llm(llm_model_id='ollama_chat/llama3.2:3b', cache=False):
    lm = dspy.LM(llm_model_id, api_base='http://localhost:11434', cache=cache)
    dspy.configure(lm=lm)

# lm = dspy.LM('ollama_chat/llama3.2:3b', api_base='http://localhost:11434')
# dspy.configure(lm=lm)

class LLMSuggestionWrapper(Wrapper):
    def __init__(self, env, llm_use_probability=0, llm_model_id='ollama_chat/llama3.2:3b', load_llm_path=None, cache_llm = False):
        super().__init__(env)
        
        # extend observation space        
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.int32)

        # setup LLM
        configure_llm(llm_model_id, cache_llm)
        self.llm_agent = dspy.ChainOfThought(GenerateAction)
        
        if load_llm_path:
            self.llm_agent.load(path=load_llm_path)
            
        
        # context sample
        ### for 1d board representation (1,-1,0)
        # self.context_sample = "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as an array of lenght 9, starting from 0 to 8. Each element represents a cell of the tictactoe board. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle."
        ### for 1d board representation (X,O,_)
        # self.context_sample = "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as an array of lenght 9, starting from 0 to 8. Each element represents a cell of the tictactoe board."
        ### for 2d board representation (1,-1,0)
        # self.context_sample = "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as a (3x3) array, with each cell being represented as an integer starting from 0 to 8. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle."
        ### for 2d board representation (X,O,_)
        # self.context_sample = "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. The board is represented as an array of lenght 9, starting from 0 to 8. Each element represents a cell of the tictactoe board."
        
        ### compatible with new GenerateAction version
        self.context_sample = "You are a bot playing a game of tic-tac-toe as a Cross (X). Your task is to play the next move and specify the grid cell index where you would place your turn. '0' indicates an empty cell, '1' indicates a cross, and '-1' indicates a circle."
        
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
        board_str, _ = self._get_board_representation(state.reshape(3, 3))
        response = self.llm_agent(context=self.context_sample, current_state=board_str)
        return int(response.answer)
    
    def _get_board_representation(self, board):
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