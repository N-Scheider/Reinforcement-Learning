import gymnasium as gym
from gymnasium import spaces
import numpy as np


class optimal_investment_env(gym.Env):
    #class wraps the optimal investment problem in a gymnasium environment
    
    def __init__(self, T=3, d=-0.5, u=2.0, q=0.5):
        super(optimal_investment_env, self).__init__()
        
        # TODO: Store the input parameters as class instance variables
        self.final_time = T
        self.down = d
        self.up = u
        self.prob = q
        
        
        #We define the observation space. It contains the information about the current timepoint, asset price and wealth
        #note that the observation space is larger than we defined in the 2nd exercise class
        #we will ensure in the step function that no state can be reached outside of the Markov Decision Model.
        self.observation_space = spaces.Dict(
                                    spaces = {
                                        "timepoint": spaces.Box(low = 0, high = T, shape = (1,), dtype = np.float64),
                                        "asset price": spaces.Box(low = 0, high = np.inf, shape = (1,), dtype = np.float64),
                                        "wealth": spaces.Box(low = 0, high = np.inf, shape = (1,), dtype = np.float64)
                                        }
                                    )
        
        #TODO define the action space
        #Hint: -Have a look at https://gymnasium.farama.org/api/spaces/
        #      -The action space should be 1-dim, representing the interval [0,\inf)  
        self.action_space = spaces.Box(low = 0, high = np.inf, shape = (1,), dtype = np.float64)
        


    def reset(self, seed = None, options = None):
        """
        Function required for a gymnasium environment. The function resets the environment to its initial state.
        If options is not None: Reset the environment to the given options
        
        Optional Inputs: - seed : sets the numpy.random.seed
                         - options (optional dict) : Implemented keys:  - "timepoint"  
                                                                        - "asset price"
                                                                        - "wealth"
                           If options is not None: reset the environment to the given state 
        """

        # TODO: reset to the initial start state or the state given in the options dictionary
        if options is None:
            #set to initial state
            timepoint = np.array([0.0])
            asset_price = np.array([8.0])
            wealth = np.array([100.0])
            # np.array([]) to be consistent with observation space
        else:
            timepoint = options["timepoint"]
            asset_price = options["asset price"]
            wealth = options["wealth"]
        
        # TODO Store the information as class instance variables
        self.timepoint = timepoint
        self.asset_price = asset_price
        self.wealth = wealth
                
        #create output dictionary
        out_dict = {"timepoint": self.timepoint,
                    "asset price": self.asset_price,
                    "wealth": self.wealth}
        info = {}
        
        return out_dict, info
    
    
    def step(self, action):
        """
        Step function, required for gymnasium environments. Calculates the next state!!! given the action
        Input: - action (np.ndarray of shape (1,)) from the action space, denotes how much money to invest in the risky asset
        Output: - out_dict (dict) information about the next state 
                - reward (float)
                - terminated, truncated (bool)
                - info (dict) not implemented.          
        """
        assert action <= self.wealth   #check that no more money is invested than the current wealth        
        assert action % self.asset_price == 0   #only integer number of assets can be bought

        # TODO: Update timepoint
        self.timepoint = self.timepoint + 1

        #TODO: sample factor change (Hint: np.random.choice)
        R = np.random.choice([self.up, self.down], p=[self.prob, 1-self.prob])
        
        self.asset_price = (1+R)*self.asset_price #TODO: Update asset price   
        self.wealth = self.wealth + action*R  #TODO: Update wealth  
        
        #TODO: calculate reward
        reward = np.log(self.wealth) if (self.timepoint == self.final_time) and self.wealth > 0 else 0 # Avoid error of logarithm with 0

            
        
        out_dict = {"timepoint": self.timepoint,
                    "asset price": self.asset_price,
                    "wealth": self.wealth}

        
        #TODO: Set the variable terminated to True when the MDP reaches timestep T
        terminated = (self.timepoint>=self.final_time)

        truncated = False
        info = {}
        return out_dict, reward, terminated, truncated, info

                
    def render(self):
        """
        Function required for gymnasium environments. We don't need it in our case.
        """
        return True