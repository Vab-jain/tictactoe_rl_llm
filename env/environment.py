# environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np



class RandomAgent:
    def __init__(self, one_hot_state = False):
        self.one_hot_state = one_hot_state
        
    def select_action(self, state): # the state here is the board.flatten() returned from _get_obs();
        if not self.one_hot_state:
            valid_actions = np.argwhere(state == 0).flatten()
            return np.random.choice(valid_actions)
        else:
            # TODO: implement one-hot state action selection for RandomAgent
            raise Exception("One-hot-state Random Action not implemented")



class TicTacToeEnv(gym.Env):
    def __init__(self,
                    reward = [1, -1, 0],  # reward = the reward for winning, losing, and drawing respectively
                    one_hot_state = False,  # one_hot_state = whether to use one-hot encoding for the state
                    env_policy = None  # env_policy = the policy that the environment will use to play the game
                                                    # env_policy is an agent with select_action() method
                    ):
        """
        Agent is always Cross (X) or (1). 
        Env is always Circle (O) or (-1). 
        """
        # Define the action space
        # Each action is a number from 0 to 8 (corresponding to positions on the 3x3 board)
        self.action_space = spaces.Discrete(9)

        # Define the observation space
        # The board is a 3x3 grid, which can have values -1 (Player 1), 1 (Player 2), and 0 (empty)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int32)

        # Initialize the game state
        self.board = None
        # define the reward for winning, losing, and drawing respectively
        self.reward_win = reward[0]
        self.reward_lose = reward[1]
        self.reward_draw = reward[2]
        # one_hot_state = whether to use one-hot encoding for the state
        self.one_hot_state = one_hot_state
        
        self.agent_player = 1
        self.env_player = -1
        
        if env_policy is None:
            self.env_policy = RandomAgent(one_hot_state=self.one_hot_state)
        else:
            self.env_policy = env_policy
            
    def reset(self, 
              first_player:int = 1,   # first_player = the player that start the game {1: self.agent_player, -1: self.env_player} 
              ):
        self.board = np.zeros((3,3), dtype=int)
        self.done = False
        self.winner = None  # 1: player-1 wins, -1: player-2 wins, 0: draw
        
        if first_player == self.env_player:
            self._env_move()  
              
        return self._get_obs()
    
    def _get_obs(self):
        '''
        Encoding of the board state
        '''
        if self.one_hot_state:
            return self._get_one_hot_obs()
        else:
            return self.board.flatten()
    
    def _get_one_hot_obs(self):
        state = np.zeros((3, 3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == self.agent_player:
                    state[i, j, 0] = 1
                elif self.board[i, j] == 0:
                    state[i, j, 1] = 1
                elif self.board[i, j] == self.env_player:
                    state[i, j, 2] = 1
        return state.flatten()
        
    def _check_winner(self):
        for player in [1, -1]:
            # Rows, columns, and diagonals
            if any(np.all(self.board[row, :] == player) for row in range(3)) \
               or any(np.all(self.board[:, col] == player) for col in range(3)) \
               or np.all(np.diag(self.board) == player) \
               or np.all(np.diag(np.fliplr(self.board)) == player):
                self.done = True    # set done to True
                self.winner = player   # set winner to the player

        if not np.any(self.board == 0):
            self.done = True  # Draw
            self.winner = 0
    
    def get_winner(self):
        return self.winner if self.winner else 0
        
    def get_available_actions(self):
        return np.argwhere(self.board.flatten() == 0).flatten()

    def _env_move(self):
        # check if any cells are empty
        available = self.get_available_actions()
        if available.size == 0:
            return
        # select action from env_policy
        env_action = self.env_policy.select_action(self._get_obs())
        row, col = divmod(env_action, 3)
        self.board[row, col] = self.env_player
        
    def step(self, action):
        if self.done:
            return self._get_obs(), 0, True
        
        # get action -> row, column
        row, col = divmod(action, 3)
        
        # check for invalid move
        assert(action in self.action_space)
        if self.board[row, col] != 0:
            # Invalid move
            reward = -10
            self.done = True
            return self._get_obs(), reward, self.done
        
        # Agent's move
        self.board[row, col] = self.agent_player
        self._check_winner()
        if self.done:
            reward = self.reward_win if self.winner == self.agent_player else self.reward_draw  # Win or draw
            return self._get_obs(), reward, self.done
        
        # Opponent's move
        self._env_move()
        self._check_winner()
        if self.done:
            reward = self.reward_lose if self.winner == self.env_player else self.reward_draw  # Loss or draw
            return self._get_obs(), reward, self.done
        
        # Continue game
        return self._get_obs(), 0, self.done
    
    def render(self):
        # Simple text-based rendering
        symbols = {1: 'X', -1: 'O', 0: ' '}
        print("\nBoard State:")
        for row in self.board:
            print('|' + '|'.join(symbols[cell] for cell in row) + '|')
