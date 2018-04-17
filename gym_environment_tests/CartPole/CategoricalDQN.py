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

def init_bins(num_bins=4,#observation_space
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
    # distrubute each elem in state to the index of the closest bin
    state = np.zeros(len(bins))
    for i in range(len(bins)):
        state[i] = np.digitize(obs[i], bins[i])
    return state
        
def get_state_as_string(state):
    return ''.join(str(int(elem)) for elem in state)

def init_Q(obs_space_int, act_space_int):
    states = []
    for i in range(10**obs_space_int):
        #populates state with left padded numbers as str
        #print(str(i).zfill(obs_space_int))
        states.append(str(i).zfill(obs_space_int))
    # TODO: custom 0000 padding on left side
        
    Q = {}
    for state in states:
        Q[state] = {}
        for action in range(act_space_int):
            Q[state][action] = 0
    return Q

def play_episode(bins, Q, act_space, epsilon=.2, viz=False):
    observation = env.reset()
    total_reward = 0
    terminal = False
    num_frames = 0

    state = get_state_as_string(to_categorical(observation, bins))

    while not terminal:
        if viz: env.render()
        if num_frames > 150: epsilon = 0
        if random.random() < epsilon:
            action = env.action_space.sample()#random.randint(0, act_space - 1)
        else:
            #max_key, max_val = max_dict(d)
            #TODO: Impliment BST to speed up max
            action = max_dict(Q[state])[0]
        
        observation, reward, terminal, info = env.step(action)
        state_next = get_state_as_string(to_categorical(observation, bins))

        total_reward += reward

        if terminal and num_frames > 150:
            reward += np.log(num_frames)
        
        if terminal and num_frames < 200:
            reward = -100

        action_next, reward_next = max_dict(Q[state_next])
        Q[state][action] += ALPHA*(reward + GAMMA * reward_next - Q[state][action])
              
        state = state_next
        num_frames += 1
        

    return total_reward, num_frames

def train(bins, act_space=None,epochs=2000, obs=False, Q=False):
    if not Q: Q = init_Q(len(bins), act_space)

    stacked_frames = []
    #TODO: Plot reward averages
    rewards = []
    for ep in range(epochs):
        epsilon = np.tanh(-ep/(epochs/2))+ 1

        ep_reward, num_frames = play_episode(bins, Q, act_space,epsilon, viz=obs)
        if ep % 100 == 0:
            print("Ep: {} | {}".format(ep, epochs),
                  "%:", round(ep*100/epochs, 2),
                  "Epsilon:", round(epsilon, 4),
                  "Avg rwd:", round(np.mean(rewards),3),
                  "Ep rwd:", ep_reward)

        stacked_frames.append(num_frames)
        rewards.append(ep_reward)

    return rewards, stacked_frames, Q

def observe_agent(Q, N=15):
    [play_episode(bins, Q, -1, viz=True) for ep in range(N)]

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
ALPHA = 0.01
GAMMA = 0.9

EPOCHS = 6000

obs_space = 4
action_space = env.action_space.n

if __name__ == "__main__":
    bins = init_bins(num_bins=obs_space,
                     depth=10, limits=[4.8, 5, 0.418, 5])

    episode_rewards, _, Q = train(bins,act_space=action_space,
                                  epochs = EPOCHS, obs=observe_training)
    
    random_rwds = [play_random() for ep in range(EPOCHS)]

    plt.title("Average Reward with Q-Learning By Episode (CartPole)")
    plot_running_avg(episode_rewards)
    plt.plot(random_rwds, color="gray", label="Random Moves Running Average")

    plt.xlabel('Training Time (episodes)', fontsize=18)
    plt.ylabel('Average Reward per Episode', fontsize=16)
    plt.legend()
    plt.show()























