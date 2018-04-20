# Categorize Continuous State Space using Binning and a simple NN
# Aggregate reward in Q-Matrix using dictionary
# \\ \\ \\ \\
# Developed by Nathan Shepherd
# Inspired by Phil Tabor
# @ https://github.com/MachineLearningLab-AI/OpenAI-Cartpole

import gym
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten

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

        self.LR = 0.05
        self.num_episodes = 0
        self.update_frequency = 100
        self.define_model(np.array(observation))

    def init_Q_matrix(self, obs):
        assert(len(obs)==self.obs_space)
        self.Q = {}
        self.find(''.join(str(int(elem)) for elem in self.digitize(obs)))

    def define_model(self, example, hidden=[128, 256, 512, 256, 64]):
        #TODO: Add regularization?
        model = Sequential()
        model.add(Dense(hidden[0], input_shape=(1,) + example.shape))
        #model.add(Flatten())

        for i in range(1, len(hidden)):
            model.add(Dense(hidden[i]))#50, 100, 50
            model.add(Activation('relu'))

        model.add(Dense(self.num_actions))
        model.add(Activation('linear'))

        adam = Adam(lr=self.LR)

        model.compile(loss='mse', optimizer=adam,
                      metrics=['accuracy'])
        print(model.summary())

        self.model = model

    def find(self, state_string):
        try:
            self.Q[state_string]
        except KeyError as e:
            self.Q[state_string] = {}
            for action in range(self.num_actions):
                self.Q[state_string][action] = 0
                
            idx = random.randint(0, self.num_actions - 1)
            self.Q[state_string][idx] = 0.1

    def get_action(self, state):        
        string_state = ''.join(str(int(elem)) for elem in self.digitize(state))
        self.find(string_state)
        return max_dict( self.Q[string_state] )[0]

    def evaluate_utility(self, state):
        string_state = ''.join(str(int(elem)) for elem in self.digitize(state))
        self.find(string_state)
        return max_dict( self.Q[string_state] )[1]

    def update_policy(self, state, state_next, action, reward):
        state_value = self.evaluate_utility(state)
        
        action = self.get_action(state)
        reward_next = self.evaluate_utility(state_next)

        state_value += ALPHA*(reward + GAMMA * reward_next - state_value)

        state = ''.join(str(int(elem)) for elem in self.digitize(state))
        self.Q[state][action] = state_value

    def preprocess(self, data):
        #print("before:", [round(d, 2) for d in data])
        # --> before: [3, 4, 3, 3, 3, 6, 8, 4]
        mean = np.mean(data)
        stddev = np.std(data)
        #standerdize state
        for i in range(len(data)):
            data[i] = (data[i] - mean) / stddev
            
        _max = max(data)
        _min = min(data)
        diff = _max-_min
        #rescale state
        for i in range(len(data)):
            data[i] = (data[i] - _min) / diff
        
        return np.array(data).flatten()

        #print("after:",[round(d, 2) for d in data],end="\n\n")
        # --> after: [0.0, 0.2, 0.0, 0.0, 0.0, 0.6, 1.0, 0.2]

    def replace_rewards(self):
        self.num_episodes += 1
        if self.num_episodes % 100 == 0:
            batch_size = len(self.Q)
            states = np.zeros((batch_size,self.obs_space))
            rewards = np.zeros((batch_size, self.num_actions))
            idx = 0
            
            print("Updating", len(self.Q),
            "state<&>reward pairs with a Neural Network")
            for state, act_dict in self.Q.items():
                #normalize states and rewards as vector
                #TODO: Try training without rescaling the reward???
                s = np.zeros((1, 1, self.obs_space))
                s[0][0] = self.preprocess([int(digit) for digit in state])

                r = np.zeros((1, 1, self.num_actions))
                r[0][0] = self.preprocess([act_dict[i] for i in range(self.num_actions)])

                state = s; reward = r;
                self.model.fit(state, reward, epochs=1, verbose=0)

            d = {}
            for k in random.sample(list(self.Q), min(len(self.Q), 1000)):
                d[k] = self.Q[k]

            scores = 0; num_s = 0;
            for state, act_dict in d.items():
                #normalize states and rewards as vector
                #TODO: Try training without rescaling the reward???
                s = np.zeros((1, 1, self.obs_space))
                s[0][0] = self.preprocess([int(digit) for digit in state])

                r = np.zeros((1, 1, self.num_actions))
                r[0][0] = self.preprocess([act_dict[i] for i in range(self.num_actions)])

                state = s; reward = r;
                scores += self.model.evaluate(state, reward, verbose=0)[-1]
                num_s += 1

                #states[idx] = state; rewards[idx] = reward
                #idx += 1
            print("Accuracy of model: {}%".format(round(scores*100/num_s, 3)))
                
            #states= np.expand_dims(states, axis = 0)
            #self.model.fit(states, rewards, batch_size=64,
            #                   epochs=1, verbose=2)
            #Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            '''
            self.model.fit(np.array(states).flatten(), np.array(rewards).flatten(), epochs=1, verbose=0,
                           batch_size = min(64, len(states)))
                           '''
                           
        

    def get_bins(self, num_bins):
        # Make 10 x state_depth matrix,  each column elem is range/10
        # Digitize using bins ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        bins = []
        ranges = [5,#~3, obs[0] == pos_x
                  10,#~5, obs[1] == pos_y
                  3,#~3, obs[2] == vel_x
                  3,#~5, obs[3] == vel_y
                  2,#~2, obs[4] == angle
                  1,#~2, obs[5] == angular_vel
                  1,#~1, obs[6] == 1 if lhs_leg_contact else 0        
                  1]#~1, obs[7] == 1 if rhs_leg_contact else 0
        for i in range(self.obs_space):
            # use minimum value to anchor buckets
            start, stop = -ranges[i], ranges[i]
            buckets = np.linspace(start, stop, num_bins)
            bins.append(buckets)
        return bins         
            
    def digitize(self, arr):
        # distrubute each elem in state to the index of the closest bin
        state = np.zeros(len(self.bins))
        for i in range(len(self.bins)):
            state[i] = np.digitize(arr[i], self.bins[i])
        return state

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

    agent.replace_rewards()

    while not terminal:
        if viz: env.render()
        #if num_frames > 300: epsilon = 0.1

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)
        
        state_next, reward, terminal, info = env.step(action)

        total_reward += reward
        
        if terminal:
            if  num_frames < 50:
                reward += 50

            # lander angular vel and pos controlled
            if abs(state[4]) < 0.1:
               if abs(state[5]) < 0.1:
                   reward += 50

            # ended with control low vertical vel
            if state[3] > 0 and state[3] < 1:
                reward += 10

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

def save_agent(A):
    with open('Agent_LunarLander_strDQN.pkl', 'wb') as writer:
        pickle.dump(A, writer, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_agent(filename):
    with open(filename, 'rb') as reader:
        unserialized_data = pickle.load(reader)


'''
    Note: String DQN is extremely sensitive to internal parameters
          Improve congvergance by solving subproblems with reward function
                                  picking a good set of range values for bins

    Highest Running Avg for StringDQN: -116 (but observe(A) showed basically solved)
    
    TODO:
    Store transition in memory lookup with state visit frequency and priority
    Every 2000 training steps:
       delete from memory all states with low visit freq and priority
    With memory at a more efficient size, train DNN on states<&>actions
    Update all actions with predictions from the DNN
    Continue Q-Learning with the new policy
    repeat

    
'''
env = gym.make('LunarLander-v2')
#env = gym.make('CartPole-v0')
observe_training = False
EPSILON_MIN = 0.2
NUM_BINS = 8#must be even#
ALPHA = 0.01
GAMMA = 0.9

EPOCHS = 100000

obs_space = 8
action_space = env.action_space.n

if __name__ == "__main__":
    episode_rewards, num_frames, Agent = train(obs_space, act_space=action_space,
                                      epochs = EPOCHS, obs = observe_training)
    print("Completed Training")
    random_rwds = []
    for ep in range(EPOCHS):
        pass# The upper bound on random LunarLander is 0
        #random_rwds.append(play_random())

    plt.title("Average Reward with Q-Learning By Episode (LunarLander)")
    plot_running_avg(episode_rewards)
    #plt.plot(random_rwds, color="gray", label="Random Moves Running Average")

    plt.xlabel('Training Time (episodes)', fontsize=18)
    plt.ylabel('Average Reward per Episode', fontsize=16)
    plt.legend()
    plt.show()

    recent_agent = 'Agent_LunarLander_strDQN.pkl'























