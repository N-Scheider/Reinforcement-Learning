#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:19:36 2025

@author: noahscheider
"""

import numpy as np
import matplotlib.pyplot as plt



# admissible actions for every state
def get_admissible_actions(p, w):
    
    max_num_assets = int(np.floor(w/p))
    
    return [p * i for i in range(max_num_assets + 1)]


# return next state, basd on (state, action) pair
def next_state(state, action, R):
            
    # Return new state
    return (state[0] + 1, state[1] * (1+R), state[2] + R * action)



# define state space
def get_all_states(start_state, T, u, d):
    all_states = [start_state]
    
    for state in all_states:
        if state[0] < T:
            for a in get_admissible_actions(state[1], state[2]):
                for R in [u, d]:
                    new_state = next_state(state, a, R)
                    all_states.append(new_state)

    return all_states



# initialize "naive control" function q:(state x action) -> R
def Initialize_Dictionary(states, random):
            
    # initialize "naive control" function q and learning rate alpha
    Dict = {}
    if random == True:    
        for state in states:
            Dict[state] = {action: np.random.rand() for action in get_admissible_actions(state[1], state[2]) }
    else:
        for state in states:
            Dict[state] = {action: np.random.rand() for action in get_admissible_actions(state[1], state[2]) }
    
    return Dict
            


# Define behavioural policy that returns an action depending on greediness of policy
def policy_behavioural(state, q, eps):
    
    # Avoid price > wealth
    if state[1] > state[2]:
        return q[state][0], 0
    
    
    # Otherwise price <= wealth
    q_state_dict = q[state]
    
    # extract admissible actions into a list
    admissible_actions = list(q_state_dict.keys())
    q_state_values = list(q_state_dict.values())

    # get optimal action
    opt_q = max(q_state_values)
    opt_a = admissible_actions[q_state_values.index(opt_q)] # when several argmax, first one is taken
    
    # random variable to determine which action follows from the state
    P = np.random.binomial(1, (1-eps)+eps/len(admissible_actions))
    admissible_actions.remove(opt_a)
    policy_action = P*opt_a + (1-P)*np.random.choice(admissible_actions)
    
    return opt_q, policy_action



def VisualizeResult(q_function, T, u ,d, q, M, eps):
    
    plot_states = []
    plot_states_opt = []
    min_wealth = + np.inf
    max_wealth = - np.inf

    for state in q_function:
        
        state_wealth = state[2]
        q_function_state_dict = q_function[state]
        
        if state[0]<T:
            
            q_state_values = list(q_function_state_dict.values())
            admissible_actions = list(q_function_state_dict.keys())
            opt_a = admissible_actions[q_state_values.index(max(q_state_values))] # when several maximizing args, first one is taken
            
            plot_states_opt.append((state_wealth, opt_a))
            
            for action in q_function_state_dict:
                
                plot_states.append((state_wealth, action))
                if min_wealth > state_wealth:
                    min_wealth = state_wealth
                if max_wealth < state_wealth:
                    max_wealth = state_wealth
                
    fig, ax = plt.subplots()
    ax.scatter(*zip(*plot_states), c='#0065bd')
    ax.scatter(*zip(*plot_states_opt), c='#F7811E')
    opt_frac = -q/d-(1-q)/u
    ax.plot([min_wealth, max_wealth], [opt_frac*min_wealth, opt_frac*max_wealth], c='black')
    ax.set_title(f"Iterations: {M}, eps {eps}")
    plt.xlabel("Total Wealth")
    plt.ylabel("Wealth in Financial Asset")

    filename = f"opt_policy_iter_{M}_eps_{eps}".replace(".", "_").replace("/", "_")
    plt.savefig(filename)

    plt.show()



"""
# compute value function for every state via action value function * policy
V = {}
for state in states:
    V[state] = 0
    admissible_actions = 0
    opt_q = -np.inf
    #opt_a = -np.inf not needed
            # gather information about the admissible action as policy will need to draw from them later
    q_function_state = q_function[state]
    for action in q_function_state:
        V[state] += q_function_state[action]*eps
    
    opt_q = max(list(q_function_state.values()))
    V[state] = V[state]/len(q_function[state]) + (1-eps)*opt_q
"""
