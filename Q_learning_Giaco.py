# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 12:47:31 2025
 The code is based on the Jupyter notebooks 4, 
@author: Giacomo Vescovi
"""

# importing libraries ############################### 

import matplotlib.pyplot as plt
import numpy as np
import random
import operator



# parameters ########################################

T     = 3     #Time horizon
u     = 2.0   #CRR parameter, up factor change
d     = -.5   #CRR parameter, down factor change
q     = .5    #CRR parameter, probability of up change
gamma = .95
start_state= (0., 8., 100.)



# function to compute all admissible actions #########

def get_admissible_actions(p, w):
    
    max_num_assets = int(np.floor(w/p))
    
    
    return [p * i for i in range(max_num_assets + 1)]



# we compute all possible states #####################

def get_all_states():
    
    states = [start_state]
    
    
    
    for state in states:
        if state[0] < T:
            for a in get_admissible_actions(state[1],state[2]):
                for R in [d,u]:
                    
                    new_state = ( state[0] + 1,
                                  state[1] * (1+R),
                                  state[2] + R * a
                                )
                    states.append(new_state)
                    
    
    states = list(set(states))

    return states


# we compute the step function ##########################

def step(state, action): #state = [time, price, wealth]
    time   = state[0]
    price  = state[1]
    wealth = state[2]
    assert action <= wealth   #check that no more money is invested than the current wealth        
    assert action %  price  == 0   #only integer number of assets can be bought
    
    
    time = time + 1 

    R = np.random.choice((d,u), p=(1-q,q))   
    
    price = price * (1+R)    
    wealth = wealth + action * R  
    
     
    reward = 0
    if time == T:
        reward = np.log(wealth)

        
    
    
    return (time, price, wealth), reward


# we write the Q-learning update rule ########################################

def Q_learning_update ( current_state, reward, next_state, Q, learning_rate, action):
    
    Q[current_state][action]= Q[current_state, action] + learning_rate * \
        (reward + gamma*np.max(Q[next_state])-Q[current_state][action])
        
    return Q


# we write the Q-learning algorithm ##########################################

def Q_learning(iterations):
    states= get_all_states()
    """
    q_table = {}
    for state in states:
        actions = get_admissible_actions(p = state[1], w = state[2])
        for action in actions:
            q_table.append({state:})
            
  """          
    
    q_table = {state: {action: np.random.rand() for action in get_admissible_actions(state[1], state[2]) } for state in states}
    #print(q_table)

    epsilon = 0.2
    
    for i in range(iterations):
        state = start_state
        
        if random.uniform(0,1)<epsilon:
            action = random.choice(get_admissible_actions(state[1], state[2]))
        else:
            action = max(q_table[state].items(), key = operator.itemgetter(1))[0]

        
        for t in range(T):
            
            next_state, reward = step(state, action)
            
            learning_rate = 0.2
            next_best_action = max(q_table[next_state].items(), key = operator.itemgetter(1) )[0]
            
            
            q_learning_target = reward + gamma * q_table[next_state][next_best_action]
            
            q_table[state][action] += learning_rate * (q_learning_target-q_table[state][action])
            
            state = next_state
            action = next_best_action
            
    return q_table
            
# we write a function to graph the optimal investement strategy###############

def optimal_investment_plotting(iterations):
    q_table = Q_learning(iterations)
    plot_states = []
    plot_states_opt = []
    min_wealth = + np.inf
    max_wealth = - np.inf
    states= get_all_states()
    
    for state in q_table:
        
        if state[0]<T:
            
            best_action = max(q_table[state].items(), key = operator.itemgetter(1))[0]
            plot_states_opt.append((state[2], best_action))
            
            for action in q_table[state]:
                
                plot_states.append((state[2],action))
                if min_wealth > state[2]:
                    min_wealth = state[2]
                if max_wealth < state[2]:
                    max_wealth = state[2]
                
    fig, ax = plt.subplots()
    ax.scatter(*zip(*plot_states), c='#0065bd')
    ax.scatter(*zip(*plot_states_opt), c='#F7811E')
    opt_frac = -q/d-(1-q)/u
    ax.plot([min_wealth, max_wealth], [opt_frac*min_wealth, opt_frac*max_wealth], c='black')
    ax.set_title(f"Iterations: {iterations}")
    plt.xlabel("Total Wealth")
    plt.ylabel("Wealth in Financial Asset")
    plt.savefig("opt_policy")
    plt.show()
                
                


optimal_investment_plotting(20000)
            
            


