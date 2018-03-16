#Solve the Lunar Lander Game enviornment

import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

LR = 1e-2
env = gym.make("LunarLander-v2")
env.reset()
goal_steps = 2000
score_requirement = -150
initial_games = 1000#10000

action_dims = 4


def create_random_samples():
    # [observations, actions]
    training_data = []
    scores = []
    # just the scores that met threshold:
    accepted_scores = []
    
    for i in range(initial_games):
        if i % 10 == 0:
            print('Completed generating samples: {}%'.format(i*100/initial_games))
            
        score = 0
        game_memory = []
        prev_observation = []

        for _ in range(goal_steps):
            action = random.randrange(0, action_dims)
            observation, reward, done, info = env.step(action)
            
            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            
            score+=reward
            if done: break

        # IF our score >= threshold, we'd like to save [action, obs] pairs
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                output = np.zeros(action_dims)
                # convert to one-hot (this is the output layer for our neural network)
                output[data[1]] = 1
                    
                training_data.append([data[0], output])
                
        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('lunar_lander_training_data.npy',training_data_save)
    
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data

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

training_data = create_random_samples()
model = train_model(training_data)

scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
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

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),
                                        choices.count(0)/len(choices)))
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
