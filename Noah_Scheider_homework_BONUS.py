#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:33:12 2025

@author: noahscheider
"""

import numpy as np
import matplotlib.pyplot as plt

from Noah_Scheider_homework_BONUS_functions import *


# FIX random seed
np.random.seed(10)


#We define the start state, which will be used in some test functions.
start_state = (0., 8., 100.)

#market parameters
T = 3     #Time horizon
u = 2.0   #CRR parameter, up factor change, slightly different from the other exercises!
d = -.5   #CRR parameter, down factor change
q = .5    #CRR parameter, probability of up change
gamma = 1 # ADDED PARAMETER BUT NOT GIVEN IN EXERCISE

# get list of all states
states = get_all_states(start_state, T, u, d)

# remove duplicates
states = list(set(states))

# Q-Learning parameters
eps = 0.8 # small epsilon greedy
M = 50000 # amount of episodes

# Initialize Q function and learning rate as dictionaries with all (state, action) pairs as keys with constant value
q_function = Initialize_Dictionary(states, random = 0)
learning_rate = Initialize_Dictionary(states, random = 1) # 1 everywhere
learning_rate_visits = Initialize_Dictionary(states, random = 0) # 0 everywhere



#### Q-Learning #####

# loop over episodes
for i in range(M):
    
    # Choose A_0^m depending on S_0^m = start_state
    state = start_state # define state as start_state to be consistent
    state_action = policy_behavioural(state, q_function, eps)[1]
    
    # loop over the length of each episode
    for n in range(T+1):
                
        # Compute necessary parameters for q estimation if        
        if n < T:
            # Initialize next step of the rollout
            R = np.random.choice([u, d], p=[q,(1-q)]) # up-down movement of CRR-Model

            new_state = next_state(state, state_action, R) # new state by oracle
            opt_q, new_state_action = policy_behavioural(new_state, q_function, eps)  # choose action from policy
        else:
            opt_q = 0
            
        q_eval = q_function[state][state_action]
        alpha = learning_rate[state][state_action]
        reward = np.log(state[2]) if state[0] == T else 0

        # Estimate q function and update learning rate        
        q_function[state][state_action] = q_eval + alpha*(reward + gamma * opt_q - q_eval)
        learning_rate_visits[state][state_action] += 1
        learning_rate[state][state_action] = 1/(1+learning_rate_visits[state][state_action])
        
        # break if terminal state
        if state[0] == T:
            break

        # Transmit values into next loop iteration
        state = new_state
        state_action = new_state_action
        

# Visualize Result
VisualizeResult(q_function, T, u ,d, q, M, eps)


