"""
This program was initialized by MorvanZhou and has been modified
--> Original can be found on his GitHub https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/experiments/Solve_LunarLander/run_LunarLander.py

Deep Q network,

LunarLander-v2 example

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from gym import wrappers
from DuelingDQNPrioritizedReplay import DuelingDQNPrioritizedReplay

env = gym.make('LunarLander-v2')
# env = env.unwrapped
env.seed(1)

N_A = env.action_space.n
N_S = env.observation_space.shape[0]
MEMORY_CAPACITY = 50000
TARGET_REP_ITER = 2000
MAX_EPISODES = 900
E_GREEDY = 0.95
E_INCREMENT = 0.00001
GAMMA = 0.99
LR = 0.0001
BATCH_SIZE = 32
HIDDEN = [400, 400]
RENDER = True

RL = DuelingDQNPrioritizedReplay(
    n_actions=N_A,
    n_features=N_S,
    learning_rate=LR,
    e_greedy=E_GREEDY,
    reward_decay=GAMMA,
    hidden=HIDDEN,
    batch_size=BATCH_SIZE,
    replace_target_iter=TARGET_REP_ITER,
    memory_size=MEMORY_CAPACITY,
    e_greedy_increment=E_INCREMENT,)

total_steps = 0
running_r = 0
r_scale = 100
for i_episode in range(MAX_EPISODES):
    state = env.reset()  # (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground, r_leg_on_ground)
    ep_r = 0
    while True:
        if total_steps > MEMORY_CAPACITY: pass#env.render()
        action = RL.choose_action(state)
        state_next, reward, done, _ = env.step(action)
        if reward == -100: reward = -30
        reward /= r_scale

        ep_r += reward
        RL.store_transition(state, action, reward, state_next)
        if total_steps > MEMORY_CAPACITY:
            RL.learn()
        if done:
            land = '| Landed' if reward == 100/r_scale else '| ------'
            running_r = 0.99 * running_r + 0.01 * ep_r
            print('Epi: ', i_episode, land,
                  '| Epi_R: ', round(ep_r, 2),
                  '| Running_R: ', round(running_r, 2),
                  '| Epsilon: ', round(RL.epsilon, 3))
            break

        state = state_next
        total_steps += 1

