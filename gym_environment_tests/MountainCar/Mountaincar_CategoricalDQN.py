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
    max_key = ""
    for key, val in d.items():
        if val > max_val:
            max_val = val
            max_key = key
    return max_key, max_val

def to_categorical(obs, depth):
    # distrubute each elem in state to the index of the closest bin
    for i in range(len(obs)):
        if obs[i] < 0.01 and obs[i] > -0.01:
            obs[i] = 0
        else:
            obs[i] = round(obs[i], depth)
            
    return [round(num, depth) for num in obs]
        
def get_state_as_string(state):
    return ''.join(str(elem)+"|" for elem in state)[:-1]

def init_Q(obs_depth, act_space_int):
    states = []
    for i in range(-10**obs_depth, 10**obs_depth):
        #populates state with left padded numbers as str
        states.append(str(i))
        
    Q = {}
    for state in states:
        Q[state] = {}
        for action in range(act_space_int):
            Q[state][action] = 0
    return Q

def play_episode(depth, Q, act_space, epsilon=.2, viz=False):
    observation = env.reset()
    total_reward = 0
    terminal = False
    num_frames = 0

    state = get_state_as_string(to_categorical(observation, depth))

    while not terminal:
        if viz: env.render()
        
        if state not in Q:
                Q[state] = {}
                for i in range(action_space):
                    Q[state][i] = 0
                    
        if num_frames > 150: epsilon = 0
        #print(observation, state)
        if random.random() < epsilon:
            action = env.action_space.sample()#random.randint(0, act_space - 1)
        else:
            #max_key, max_val = max_dict(d)
            #TODO: Impliment BST to speed up max
                    
            action = max_dict(Q[state])[0]
        
        observation, reward, terminal, info = env.step(action)
        state_next = get_state_as_string(to_categorical(observation, depth))

        '''
        if terminal:
            print(state,"\t",observation, end="\t")
            print(int(10*observation[0]))
            reward += int(10*observation[0]) #position of cart
        '''
        if terminal and observation[0] > 0.3:
            reward += 500 + abs(25*observation[0])
        if terminal and abs(observation[0]) > 0.3:
            #print(state,"\t",observation, end="\t")
            reward += 400 + abs(10*observation[0])
            #print(reward)
            
        if terminal and abs(observation[0]) < 0.3:
            #print(state,"\t",observation, end="\t")
            reward += -100
            #print(reward, ">>>>>")

        total_reward += reward
        
        #if terminal and num_frames < 200:
        #    reward = -300
        if state_next not in Q:
            Q[state_next] = {}
            for i in range(action_space):
                Q[state_next][i] = 0
                
        action_next, reward_next = max_dict(Q[state_next])
        Q[state][action] += ALPHA*(reward + GAMMA * reward_next - Q[state][action])
              
        state = state_next
        num_frames += 1
        

    return total_reward, num_frames

def train(depth=5, act_space=None,epochs=2000, obs=False, Q=False):
    if not Q: Q = init_Q(depth, act_space)

    stacked_frames = []
    rewards = [0]
    for ep in range(epochs):
        epsilon = np.tanh(-ep/(epochs/2))+ 1

        ep_reward, num_frames = play_episode(depth, Q, act_space,epsilon, viz=obs)
        if ep % 100 == 0:
            print("Ep: {} | {}".format(ep, epochs),
                  "%:", round(ep*100/epochs, 2),
                  "Epsilon:", round(epsilon, 4),
                  "Avg rwd:", round(np.mean(rewards),3),
                  "Ep rwd:", round(ep_reward, 3))

        stacked_frames.append(num_frames)
        rewards.append(ep_reward)

    return rewards, stacked_frames, Q

def observe_agent(Q, N=15):
    [play_episode(STATE_DEPTH, Q, -1, viz=True) for ep in range(N)]

def plot_running_avg(reward_arr):
    N = len(reward_arr)
    #init unitialized array
    # (faster than np.zeros)
    running_avg = np.empty(N)

    for t in range(100, N):
        running_avg[t] = np.mean(reward_arr[t-100: t+1])

    plt.plot(running_avg, color="purple", label="Q-Learning Running Average")

def play_random():
    observation = env.reset()
    total_reward = 0
    terminal = False

    while not terminal:
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        total_reward += reward

    return total_reward

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=400,      # MountainCar-v0 uses 200
    reward_threshold=-110.0,
)
env = gym.make('MountainCarMyEasyVersion-v0')
observe_training = False
ALPHA = 0.01
GAMMA = 0.9

EPOCHS = 2000

obs_space = 2
STATE_DEPTH = 4# < 8
action_space = env.action_space.n

if __name__ == "__main__":
    #bins = init_bins(num_bins=obs_space,depth=10, limits=[2, 2])
     

    episode_rewards, _, Q = train(depth=STATE_DEPTH,act_space=action_space,
                                  epochs = EPOCHS, obs=observe_training)
    
    random_rwds = [play_random() for ep in range(EPOCHS)]

    plt.title("Average Reward with Q-Learning By Episode (CartPole)")
    plot_running_avg(episode_rewards)
    plt.plot(random_rwds, color="gray", label="Random Moves Running Average")

    plt.xlabel('Training Time (episodes)', fontsize=18)
    plt.ylabel('Average Reward per Episode', fontsize=16)
    plt.legend()
    plt.show()























