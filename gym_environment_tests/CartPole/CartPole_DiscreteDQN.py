# Categorize Continuous State Space using Binning
# Aggregate reward in Q-Matrix using dictionary
# \\ \\ \\ \\
# Developed by Nathan Shepherd
# Inspired by Phil Tabor
# @ https://github.com/MachineLearningLab-AI/OpenAI-Cartpole

import gym
import random
import numpy as np
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
    def __init__(self, obs_space, num_bins, num_actions, observation):

        self.obs_space = obs_space
        self.num_actions = num_actions
        
        self.bins = self.get_bins(num_bins)
        self.init_Q_matrix(observation)

    def init_Q_matrix(self, obs):
        assert(len(obs)==self.obs_space)
        self.Q = [{} for i in range(self.obs_space)];
        self.set_new_state(obs);

    def get_action(self, state):
        act_dict = self.find(state)
        action = self.get_max_action(act_dict)

        return action

    def get_max_action(self, nested_action_dict):
        #TODO try taking the average of actions instead of the sum

        actions = {}
        for i in range(self.num_actions):
            actions[i] = 0

        for d in nested_action_dict:
            for key, val in d.items():
                actions[key] += val/self.num_actions #np.mean([val, actions[key]])
                #actions[key] = np.tanh(val + actions[key])

        #print(actions)
        return int(max_dict(actions)[0])

    def evaluate_utility(self, state):
        action_dict = self.find(state)
        
        combined = {}
        for i in range(self.num_actions):
            combined[i] = 0

        for d in action_dict:
            for key, val in d.items():
                combined[key] += val/self.num_actions#np.mean([val, combined[key]])
                #combined[key] = np.tanh(val + combined[key])
            
        return float(max_dict(combined)[1])

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

    def get_bins(self, num_bins):
        # Make 10 x state_depth matrix,  each column elem is range/10
        # Digitize using bins ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # obs[0]--> max: 0.265506 | min: -0.149958 | std: 0.151244
        # obs[1]--> max: 0.045574 | min: -0.036371 | std: 0.032354
        # obs[2]--> max: 0.241036 | min: -0.336625 | std: 0.205835
        # obs[3]--> max: 0.046279 | min: -0.051943 | std: 0.039247        
        
        bins = []
        ranges = [0.5, 0.5, 0.5, 0.5]
        for i in range(self.obs_space):
            # use minimum value to anchor buckets
            start, stop = -ranges[i], ranges[i]
            buckets = np.linspace(start, stop, num_bins)
            bins.append(buckets)
        for b in bins:
            print(b)
        return bins
            
            
            
    def digitize(self, arr):
        for i, elem in enumerate(arr):
            idx = len(self.bins)
            for j in range(len(self.bins[i]) - 1):
                if self.bins[i][j] < elem and elem < self.bins[i][j + 1]:
                    idx = j; break;
            arr[i] = idx
        return arr
        
    def set_new_state(self, obs_sequence):
        assert(len(obs_sequence) == self.obs_space)
        obs_sequence = self.digitize(obs_sequence)
        
        init_actions = {}
        for i in range(self.num_actions):
            init_actions[i] = random.random()/10
            
        for i in range(self.obs_space):
            self.Q[i][obs_sequence[i]] = init_actions
            
    def get_nested(self, path):
        assert(len(path) == self.obs_space)
        path = self.digitize(path)
        nested_actions = []
        for i in range(len(path)):
            try:
                nested_actions.append(self.Q[i][path[i]])
            except KeyError as e:
                self.set_new_state(path)
                nested_actions.append(self.Q[i][path[i]])
        return nested_actions
                            
    def set_nested(self, path, action, value):
        assert(len(path) == self.obs_space)
        path = self.digitize(path)
        self.find(path)
        for i in range(len(path)):
            self.Q[i][path[i]][action] = value

    def get_state_stats(self):
        for i in range(len(self.Q)):
            print("\nElem:",i,end=" ")
            keys = [key for key in self.Q[i].keys()]
            print("Range: [%s, %s]" % (min(keys), max(keys)),
                  "STDDEV:", round(np.std(keys), 3), "Count:" , len(keys))


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
            #if num_frames > 150:
            #    reward += np.log(num_frames)
        
            if  num_frames < 200:
                reward = -300

        action_next = agent.get_action(state_next)
        
        agent.update_policy(state, state_next, action, reward)
              
        state = state_next
        num_frames += 1
        

    return total_reward, num_frames

def train(obs_space, act_space=None,epochs=2000, obs=False, agent=False):
    if not agent: agent = DQN(obs_space, NUM_BINS, act_space, env.reset())

    stacked_frames = []
    #TODO: Plot reward averages
    rewards = [0]
    for ep in range(epochs):
        epsilon = max(EPSILON_MIN, np.tanh(-ep/(epochs/2))+ 1)

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

gym.envs.register(
    id='CartPoleExtraLong-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=250,
    reward_threshold=-110.0,
)
env = gym.make('CartPoleExtraLong-v0')
#env = gym.make('CartPole-v0')
observe_training = False
EPSILON_MIN = 0.1
NUM_BINS = 40
ALPHA = 0.01
GAMMA = 0.9
'''
    TODO: Fix the Q matrix s.t. it creates the necissary bins
          Try concatinating bin indecies as string to correlate states
'''

EPOCHS = 6000

obs_space = 4
action_space = env.action_space.n

if __name__ == "__main__":
    episode_rewards, _, Agent = train(obs_space, act_space=action_space,
                                      epochs = EPOCHS, obs = observe_training)
    
    random_rwds = [play_random() for ep in range(EPOCHS)]

    plt.title("Average Reward with Q-Learning By Episode (CartPole)")
    plot_running_avg(episode_rewards)
    #plt.plot(random_rwds, color="gray", label="Random Moves Running Average")

    plt.xlabel('Training Time (episodes)', fontsize=18)
    plt.ylabel('Average Reward per Episode', fontsize=16)
    plt.legend()
    plt.show()























