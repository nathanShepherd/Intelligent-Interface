#Solve the Lunar Lander Game enviornment

import gym
import random
import numpy as np

from statistics import median, mean
from collections import Counter

import sys
try:
    from DDQN_April_10 import DQN
except ImportError as e:
    print(e)
    sys.path.append('./../../')
    #from CustomDQN_March_18 import DQN

def create_random_samples(init_obs):
    # [state, action, reward, state_new, done]
    training_data = []
    scores = []

    # just the scores that met threshold:
    accepted_scores = []

    for i in range(initial_games):
        if i % 10 == 0:
            print('Observing random samples: {}%'.format(i*100/initial_games))

        score = 0
        game_memory = []
        state = init_obs

        for _ in range(goal_steps):
            action = random.randrange(0, action_space)
            if action == action_space:
                print("FOUND")
            #print("ACTION::::", action, type(action))
            state_new, reward, done, info = env.step(action)
            #env.render()
            '''
            #if Lander exceeds height of game screen, end game and add penelty
            if state_new[1] > 600:
                reward -= 200
            '''

            #squash reward
            reward /= reward_discount

            game_memory.append([state, action, reward, state_new, done])
            score += reward

            if done: break

        # IF our score >= threshold, we'd like to save [action, obs] pairs
        if score >= score_requirement:
            accepted_scores.append(score)

            for data in game_memory:
                training_data.append(data)

        init_obs = env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    #np.save('lunar_lander_training_data.npy',training_data_save)

    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print("Number of acccepted scores:", len(accepted_scores))

    return np.array(training_data)

def observe_agent(A, env):
    # Observe Agent after training
    for each_game in range(15):
        state = env.reset()
        for episode in range(goal_steps):
            env.render()

            #action = Agent.use_policy(state)
            action = A.get_action(state)

            state_new, reward, done, info = env.step(action)
            #Agent.store_transition(state, action, reward, state_new, done)
            state = state_new
            if done: break

#~~~~~~~~~~~[  MAIN  ]~~~~~~~~~~~#

env = gym.make("CartPole-v0")
goal_steps = 2000#2000
score_requirement = -1.5#-1.5
initial_games = 250#10000, 250

num_training_games = 1000#>1000
action_space = 2

reward_discount = 1

observe_training = False

'''
Agent attains full control of the LunarLander
    after ~650 iterations. However,  Agent fails
    to discover the greatest reward resulting from landing
    within the flags. Instead, the Agent avoids
    tipping over and touching the ground.
    
Parameters: Epsilon_min == 0.2
            learning_rate=0.0005
            replace_target_iter=2000
            hidden= [50, 50, 100, 25]
    
'''
#TODO: Pickle memory and save episode reward

if __name__ == "__main__":
    Agent = DQN(batch_size=64,#64
                memory_size=50000,
                learning_rate=0.05,
                replace_target_iter=1000,#2000
                
                random_numerator=1000,
                random_action_decay=0.99,
                random_action_chance=0.1,
                )
                #epsilon_max=0.9,
                #random_action_chance=0.2)

    training_data = create_random_samples(env.reset())
    print("Storing random games data")
    for datum in training_data:
        s, a, r, s_, done = datum
        Agent.store_transition(s, a, r, s_, done)

    Agent.init_model(training_data[0][0].shape, action_space,
                     hidden= [64])
    #Agent.load_model("./../../saved_models/CartPole/Apr-10_avg_score~0.20834.h5")
    #observe_agent(Agent, env)
    #Agent.train()
    
    score_length = 100000
    scores = []
    avg_score = 0
    for each_game in range(num_training_games):
        #sample state from env
        #State shape: (1, 1, 8)
        state = env.reset()

        total_reward = 0
        for episode in range(goal_steps):
            if observe_training: env.render()

            #ACTION:::: 3
            action = Agent.get_action(state)
            #print("ACTION::::", action)

            state_new, reward, done, info = env.step(action)

            #squash reward
            reward /= reward_discount

            Agent.store_transition(state, action, reward, state_new, done)
            #if Agent.memory.is_full(): Agent.train()
            Agent.train()

            total_reward += reward
            state = state_new
            if done: break

        scores.append(total_reward)
        avg_score = (avg_score + total_reward)/2.0

        if each_game % 1 == 0:
            if len(scores) > score_length:
                scores = scores[-1000:]
            print("Ep: {} | {}".format(each_game, num_training_games),
                  "%:", round(each_game*100/num_training_games, 2),
                  "Epsilon:", round(Agent.epsilon, 5),
                  "avg frame rwd:",round(mean(Agent.memory.get_scores()), 3),
                  "avg ep rwd:", round(avg_score, 3))


    observe_agent(Agent, env)

    Agent.save_model("./../../saved_models/CartPole/")
    Agent.display_statisics_to_console()
    print("Score Requirement:",score_requirement)
    Agent.memory.graph()
