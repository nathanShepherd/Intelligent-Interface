# Categorize Continuous State Space using Binning
# Aggregate reward in Q-Matrix using dictionary
# \\ \\ \\ \\
# Developed by Nathan Shepherd
# Inspired by Phil Tabor
# @ https://github.com/MachineLearningLab-AI/OpenAI-Cartpole

import gym
import random
import numpy as np
from scipy import stats
from functools import reduce
import matplotlib.pyplot as plt

def max_dict(d):
    max_val = float('-inf')
    max_key = ""
    for key, val in d.items():
        if val > max_val:
            max_val = val
            max_key = key
    return max_key, max_val

class DQN:
    def __init__(self, state_depth, obs_space, num_actions, observation):
        self.obs_space = obs_space
        self.state_depth = state_depth

        self.num_actions = num_actions
        self.init_Q_matrix(observation)

    def init_Q_matrix(self, obs):
        assert(len(obs)==self.obs_space)
        self.Q = [{} for i in range(self.obs_space)];
        self.set_new_state(obs);

    def get_action(self, state):
        act_dict = self.find(state)
        action = self.get_mode(act_dict)

        return action

    def get_mode(self, nested_action_dict):
        pred_actions = []
        for i in range(self.obs_space):
            pred_actions.append( int(max_dict(nested_action_dict[i])[0]) )
        return int(stats.mode(pred_actions)[0])

    def evaluate_utility(self, state):
        action_dict = self.find(state)
        values = []
        for i in range(self.obs_space):
            values.append(float(max_dict(action_dict[i])[1]))
            
        return float(np.mean(values))

    def find(self, state):
        try:
            act_dict = self.get_nested(state)
        except KeyError as e:
            self.set_new_state(state)
            act_dict = self.get_nested(state)
        return act_dict


    def update_policy(self, state, state_next, action, reward):        
        state_value = self.evaluate_utility(state)
        
        action = self.get_action(state)
        reward_next = self.evaluate_utility(state_next)

        state_value += ALPHA*(reward + GAMMA * reward_next - state_value)
        
        self.set_nested(state, action, state_value)

    def digitize(self, arr):
        return list(map(lambda x: round(x, self.state_depth), arr))
        
    def set_new_state(self, obs_sequence):
        assert(len(obs_sequence) == self.obs_space)
        obs_sequence = self.digitize(obs_sequence)
        
        init_actions = {}
        for i in range(self.num_actions):
            init_actions[i] = 0#random.random()/10
        for i in range(self.obs_space):
            self.Q[i][obs_sequence[i]] = init_actions
            
    def get_nested(self, path):
        assert(len(path) == self.obs_space)
        path = self.digitize(path)
        nested_actions = []
        for i in range(len(path)):
            nested_actions.append(self.Q[i][path[i]])
        return nested_actions
                            
    def set_nested(self, path, action, value):
        assert(len(path) == self.obs_space)
        path = self.digitize(path)
        self.find(path)
        for i in range(len(path)):
            self.Q[i][path[i]][action] = value

            

def play_episode(agent, act_space, epsilon=.2, viz=False):
    state = env.reset()
    total_reward = 0
    terminal = False
    num_frames = 0

    max_rwd = -200
    while not terminal:
        if viz: env.render()
        #if num_frames > 300: epsilon = 0.1

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            #max_key, max_val = max_dict(d)
            #TODO: Impliment BST to speed up max
            action = agent.get_action(state)
        
        state_next, reward, terminal, info = env.step(action)

        total_reward += reward
        
        if terminal:
            if num_frames > 150:
                reward += np.log(num_frames)

            # lander angular vel and pos controlled
            #if abs(observation[4]) < 0.1:
            #   if abs(observation[5]) < 0.1:
            #       reward += 50

            # ended with control low vertical vel
            if state[3] > 0 and state[3] < 1:
                reward += 10

            if reward < (1*max_rwd)/3:
                reward += -300
                
            if reward >= max_rwd:
                max_rwd = reward
                reward += 300

        action_next = agent.get_action(state_next)
        
        agent.update_policy(state, state_next, action, reward)
              
        state = state_next
        num_frames += 1
        

    return total_reward, num_frames

def train(state_depth, obs_space, act_space=None,epochs=2000, obs=False, agent=False):
    if not agent: agent = DQN(state_depth, obs_space, act_space, env.reset())

    stacked_frames = []
    #TODO: Plot reward averages
    rewards = [0]
    for ep in range(epochs):
        epsilon = max(EPSILON_MIN, np.tanh(-ep/(epochs*2))+ 1)

        ep_reward, num_frames = play_episode(agent, act_space, epsilon, viz=obs)
        if ep % 100 == 0:
            print("Ep: {} | {}".format(ep, epochs),
                  "%:", round(ep*100/epochs, 2),
                  "Epsilon:", round(epsilon, 4),
                  "Avg rwd:", round(np.mean(rewards),3),
                  "Ep rwd:", round(ep_reward, 3))

        stacked_frames.append(num_frames)
        rewards.append(ep_reward)

    return rewards, stacked_frames, agent

def observe(agent, N=15):
    [play_episode(agent, -1, viz=True) for ep in range(N)]

def plot_running_avg(reward_arr):
    N = len(reward_arr)
    #init unitialized array
    # (faster than np.zeros)
    running_avg = np.empty(N)

    for t in range(100, N):
        running_avg[t] = np.mean(reward_arr[t-100: t+1])

    plt.plot(running_avg, color="purple", label="Q-Learning Running Average")

def play_random(viz=False):
    observation = env.reset()
    total_reward = 0
    terminal = False

    while not terminal:
        if viz: env.render()
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        total_reward += reward

        #if terminal and num_frames < 200:
         #   reward = -300
        
    return total_reward

env = gym.make('LunarLander-v2')
observe_training = False
EPSILON_MIN = 0.1
ALPHA = 0.01
GAMMA = 0.9

EPOCHS = 10**5
state_depth = 2
#depth of state rounding
# if depth == 3
# then [0.12345] -> [0.123]

obs_space = 8
# obs[0] == pos_x
# obs[1] == pos_y
# obs[2] == vel_x
# obs[3] == vel_y
# obs[4] == angle
# obs[5] == angular_vel
# obs[6] == 1 if lhs_leg_contact else 0
# obs[7] == 1 if rhs_leg_contact else 0
action_space = env.action_space.n#4

'''
Arranged observations in a tree, state is float
obs[0] -> actions[:n]
obs[2] -> actions[:n]
obs[3] -> actions[:n]
To choose an action, take mode(max(action for all observations)))

TODO:<><><><><><><><><><><><><><><><><><><>
    take actions based on max sum of actions for each state dictionary
    
    try taking the same action for multiple
        timesteps (3-5?) then observing the
          state to choose the next actions
'''

if __name__ == "__main__":

    episode_rewards, _, Agent = train(state_depth, obs_space, act_space=action_space,
                                  epochs = EPOCHS, obs=observe_training)
    
    #random_rwds = [play_random() for ep in range(EPOCHS)]

    plt.title("Average Reward with Q-Learning By Episode (LunarLander)")
    plot_running_avg(episode_rewards)
    #plt.plot(random_rwds, color="gray", label="Random Moves Running Average")

    plt.xlabel('Training Time (episodes)', fontsize=18)
    plt.ylabel('Average Reward per Episode', fontsize=16)
    plt.legend()
    plt.show()























