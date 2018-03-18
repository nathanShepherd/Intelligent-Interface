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

LR = 1e-2
env = gym.make("LunarLander-v2")
goal_steps = 90#2000
score_requirement = -150#TODO: determine good value
initial_games = 100#10000

action_space = 3

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
                '''
                action_vect = np.zeros(action_space)
                # convert to one-hot (this is the output layer for our neural network)
                action_vect[data[1]] = 1
                # [state, action, reward, state_new, done]
                training_data.append([(action_vect if i == 1 else elem) for i, elem in enumerate(data)])
                '''
                
        init_obs = env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    #np.save('lunar_lander_training_data.npy',training_data_save)
    
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return np.array(training_data)

def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.9)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.9)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.9)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.9)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.9)

    network = fully_connected(network, action_dims, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=3,
              snapshot_step=500, show_metric=True, run_id='openai_learning')
    return model

if __name__ == "__main__":
    Agent = DQN()
    
    training_data = create_random_samples(env.reset())
    for datum in training_data:
        s, a, r, s_, done = datum
        Agent.store_transition(s, a, r, s_, done)

    Agent.init_model(training_data[0][0].shape, action_space)
    Agent.siraj_train()
    #model = train_model(training_data)

    scores = []
    choices = []
    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        state = env.reset()

        for episode in range(goal_steps):
            env.render()
            action = Agent.get_action(state)
            #print("ACTION::::", action)
            
            
            state_new, reward, done, info = env.step(action)
                
            Agent.store_transition(state, action, reward, state_new, done)
            
            scores.append(reward)
            state = state_new
            
            if done: break
        
        '''
        for _ in range(goal_steps):
            env.render()

            if len(prev_obs)==0:
                action = random.randrange(0,action_dims)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

            choices.append(action)
                
            new_observation, reward, done, info = env.step(action)
        
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break
        '''

        scores.append(score)

    print('Average Score:',sum(scores)/len(scores))
    #print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    print(score_requirement)


'''
def some_random_games_first():
    # Each of these is its own game.
    for episode in range(5):
        env.reset()
        print("\nactions: ")
        # this is each frame, up to 200...but we wont make it that far.
        for t in range(200):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()
            print(action, end="")
            
            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
           
            
            if done:
                break

create_random_samples()
#some_random_games_first()
'''
