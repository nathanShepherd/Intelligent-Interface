#Solve the Lunar Lander Game enviornment

import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

from CustomDQN import DQN

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
    print(Counter(accepted_scores))
    
    return np.array(training_data)

#~~~~~~~~~~~[  MAIN  ]~~~~~~~~~~~#

LR = 1e-2
env = gym.make("LunarLander-v2")
goal_steps = 1000#2000
score_requirement = -150#TODO: determine good value
initial_games = 500#10000

num_training_games = 1000
action_space = 4

if __name__ == "__main__":
    Agent = DQN(batch_size=250,#64
                memory_size=50000,
                learning_rate=0.005)
    
    training_data = create_random_samples(env.reset())
    for datum in training_data:
        s, a, r, s_, done = datum
        Agent.store_transition(s, a, r, s_, done)

    Agent.init_model(training_data[0][0].shape, action_space)
    Agent.train()

    score_length = 1000
    scores = []
    for each_game in range(num_training_games):
        #sample state from env
        #State shape: (1, 1, 8)
        state = env.reset()

        total_reward = 0
        for episode in range(goal_steps):
            #env.render()

            #ACTION:::: 3
            action = Agent.get_action(state)
            #print("ACTION::::", action)
            
            
            state_new, reward, done, info = env.step(action)

                
            Agent.store_transition(state, action, reward, state_new, done)

            total_reward += reward
            state = state_new
            if done: break
        
        scores.append(total_reward)

        if each_game % 10 == 0:
            if len(scores) > 1000:
                scores = scores[-1000:]
            print("Percent done:", each_game*100/num_training_games,
                  "mean:",mean(scores), "last 10 reward:", mean(scores[-10:]))
        Agent.train()

    # Observe Agent after training
    for each_game in range(10):
        state = env.reset()
        for episode in range(goal_steps):
            env.render()
            action = Agent.get_action(state)
            state_new, reward, done, info = env.step(action)
            Agent.store_transition(state, action, reward, state_new, done)      
            state = state_new
            if done: break

    Agent.display_statisics_to_console()
    print("Score Requirement:",score_requirement)
















