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
        # Elem: 0 Range: [-0.50, 0.75] STDDEV: 0.32 Count: 106
        # Elem: 1 Range: [-1.97, 3.05] STDDEV: 1.10 Count: 299
        # Elem: 2 Range: [-0.27, 0.26] STDDEV: 0.16 Count: 54
        # Elem: 3 Range: [-3.32, 3.04] STDDEV: 1.56 Count: 533        
        
        bins = []
        ranges = [5, 5, 0.5, 5]
        for i in range(self.obs_space):
            # use minimum value to anchor buckets
            start, stop = -ranges[i], ranges[i]
            buckets = np.linspace(start, stop, num_bins)
            bins.append(buckets)
##        for i, b in enumerate(bins):
##            print('\t',i)
##            for num in b:
##                print(num)
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


def print_d(lexio):
    for d in lexio:
        print("")
        for key, act_d in d.items():
            print(key, end="")
            for val in act_d:
                print('\t', val)

def get_obs_samples():
    sampl = []
    for i in range(OBS_SPACE*10):
        sampl.append([i/10, i/5])
    return sampl

env = gym.make('CartPole-v0')
def get_obs(num_games, viz=False):
    observations = []
    total_reward = 0
    terminal = False
    for i in range(num_games):
        terminal = False;env.reset()
        while not terminal:
            if viz: env.render()
            action = env.action_space.sample()
            obs, reward, terminal, info = env.step(action)
            
            observations.append(obs)
    return observations
        
if __name__ == "__main__":
    OBS_SPACE= 4
    NUM_BINS= 10
    NUM_ACTS=  2
    
    a = DQN(OBS_SPACE, NUM_BINS, NUM_ACTS, env.reset())
    arr = get_obs(10000)
    w = [arr[0] for i in range(len(arr))]
    x= [arr[1] for i in range(len(arr))]
    y= [arr[2] for i in range(len(arr))]
    z= [arr[3] for i in range(len(arr))]
    print("obs[0]--> max: %f | min: %f | std: %f"% (np.amax(w),np.amin(w),np.std(w)))
    print("obs[1]--> max: %f | min: %f | std: %f"% (np.amax(x),np.amin(x),np.std(x)))
    print("obs[2]--> max: %f | min: %f | std: %f"% (np.amax(y),np.amin(y),np.std(y)))
    print("obs[3]--> max: %f | min: %f | std: %f"% (np.amax(z),np.amin(z),np.std(z)))
            
    #plt.plot([v[0] for v in arr], color="blue", label='arr[0]')
    #plt.plot([v[1] for v in arr], color="green", label='arr[1]')
    #plt.plot([v[2] for v in arr], color="red", label='arr[2]')
    #plt.plot([v[3] for v in arr], color="purple", label='arr[3]')
    #plt.legend()
    #plt.show()
    
    























    




