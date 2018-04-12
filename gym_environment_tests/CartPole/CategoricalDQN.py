# Categorize Continuous State Space using Binning
# Aggregate reward in Q-Matrix using dictionary
# \\ \\ \\ \\
# Developed by Nathan Shepherd
# Adapted from Phil Tabor
# @ https://github.com/MachineLearningLab-AI/OpenAI-Cartpole

import gym
import random
import numpy as np
import matplotlib.pyplot as plt

def max_dict(d):
    max_val = float('-inf')
    for key, val in d.items():
        if val > max_val:
            max_val = val
            max_key = key
    return max_val, max_key

def create_bins(num_bins=4,#action_space
                depth=10, limits=[4.8, 5, 0.418, 5]):
  # obs[0] -> cart position --- -4.8 - 4.8
  # obs[1] -> cart velocity --- -inf - inf
  # obs[2] -> pole angle    --- -0.418 - 0.418
  # obs[3] -> pole velocity --- -inf - inf
  bins = np.zeros((num_bins, depth))
  for i in range(num_bins):
      bins[i] = np.linspace(-limits[i], limits[i], depth)

  return bins


def to_categorical(obs, bins):
    state = np.zeros(len(bins))
    # distrubute each elem in state to the index of the closest bin
    state[i] = [np.digitize(obs[i], bins[i]) for i in range(len(bins))]
        
def get_state_as_string(state):
    return ''.join(str(int(elem)) for elem in state)

def init_Q(obs_space_int, act_space_int):
    states = []
    for i in range(10**obs_space_int):
        #populates state with left padded numbers as str
        states.append(str(i).zfill(obs_space_int))
        
    Q = {}
    for state in states:
        Q[state] = {}
        for action in range(act_space_int):
            Q[state][action] = 0
    return Q

def play_episode(bins, Q, act_space, epsilon=.2, viz=False):
    observation = env.reset()
    total_reward = 0
    termial = False
    num_frames = 0

    stats = get_state_as_string(to_catergorical(observation, bins))

    while not terminal:
        if viz: env.render()

        if random.random() < epsilon:
            action = random.randint(0, act_space - 1)
        else:
            #max_value, max_key = max_dict(d) 
            action = max_dict(Q[state])[0]

        observation, reward, terminal, info = env.step(action)
        state_next = get_state_as_string(to_categorical(observation, bins))

        reward_next, action_next = max_dict(Q[state_next])
        #determine which is the correct equation
        #Q[state][action] += reward + gamma * reward_next - Q[state][action]
        #fill used the one below but it seems wrong ...
        #Q[state][action] += reward + gamma * action_next - Q[state][action]
        
        if terminal and num_frames < 200:
            reward = -300
            
        total_reward += reward
        state = state_next
        num_frames += 1

    return total_reward, num_frames
        
action_space = 4






















