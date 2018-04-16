import gym
#import universe

#for saving models with date information
import datetime

import random
import numpy as np

import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout, Conv2D

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.container = []

    def push_sample(self,state, action, reward, state_next, done):
        if len(self.container) > self.capacity:
            del self.container[random.randint(0, len(self.container) - 1)]

        self.container.append([state, action, reward, state_next, done])

    def get_sample(self, size):
        return random.sample(self.container, size)

    def get_scores(self):
        return [datum[2] for datum in self.container]

    def get_actions(self):
        return [datum[1] for datum in self.container]

    def get_capacity(self):
        return self.capacity

    def graph(self, num_scores=1000):
        data = self.get_scores()[-1::][:num_scores]
        print("Displaying last {} scores".format(num_scores))
        plt.plot([i for i in range(len(data))], data)
        plt.show()


class DQN:
    def __init__(
            self,
            action_space=None,
            observation_space=None,

            memory_size=10000,

            reward_decay=0.9,
            learning_rate=0.005,
            random_action_chance=0.9,
            random_action_decay=0.99,

            hidden=None,
            batch_size=32,
    ):
        self.action_space = action_space
        self.observation_space = observation_space

        self.memory_size = memory_size

        self.gamma = reward_decay
        self.LR = learning_rate
        self.epsilon = random_action_chance
        self.random_action_decay = random_action_decay

        self.hidden = None
        self.batch_size = batch_size

        self.memory = Memory(memory_size)

    def init_model(self, observation_space, nb_actions):#TODO: parameterize
        self.action_space = nb_actions
        self.observation_space = observation_space

        model = Sequential()
        #32 convolution filters of size 3x3 each.
        model.add(Conv2D(32, (3, 3),
                         activation='relu',
                         input_shape=observation_space))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dense(25))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))

        adam = Adam(lr=self.LR)

        model.compile(loss='mse',
                      optimizer=adam,
                      metrics=['accuracy'])
        print(model.summary())

        self.model = model

    def _take_random_action(self):
        if random.random() < 0.8:#TODO: Anneal click probability
            return random.randrange(0, self.action_space - 1)
        else:# Agent is very unlikely to click at first
            return self.action_space - 1

    def store_transition(self, state, action, reward, state_next, done):
        self.memory.push_sample(state, action, reward, state_next, done)

    '''less likely to randomly take action relative to get_action (improved stability)'''
    def use_policy(self, state):
        #state = np.expand_dims(state, axis=0)
        #state = np.stack((state,), axis=1)

        pred = self.model.predict(state)
        return np.argmax(pred)

    def get_action(self, state):
        # decrement epsilon via: e = e_min + (e_max - e_min)^(lambda*time)
        # where lambda is the decay factor
        if random.random() < self.epsilon:
            # (left, right, up, down, click) for Click games
            return self._take_random_action()
        else:
            #add another dimension to state (i.e. batch size is one)
            #state = np.array(state)
            #state = np.expand_dims(state, axis=0)
            #state = np.stack((state,), axis=1)
            #state = np.stack(state, axis=0)
            #print("State shape:",state.shape)
            #print("Predicting action from state")
            pred = self.model.predict(state)
            #print("model prediction:", p)
            return np.argmax(pred)

    '''Developed with help from Siraj Raval: https://youtu.be/A5eihauRQvo'''
    def train(self):
        #get training samples from memory
        transitions = self.memory.get_sample(self.batch_size)

        inputs_shape = (self.batch_size,) + self.observation_space
        inputs = np.zeros(inputs_shape)
        targets = np.zeros((self.batch_size, self.action_space))

        #decrease random action probability
        self.epsilon *= self.random_action_decay

        #print("Training on batch of size:", self.batch_size)
        for i in range(len(transitions)):
            #print("Making target: {} of {}".format(i, len(transitions)))
            trans = transitions[i]
            state = trans[0];
            action = trans[1];
            reward = trans[2];
            state_new = trans[3]
            done = trans[4]


            targets[i] = self.get_action(state)
            #print("Target {}: {}".format(i, targets[i]))#Target 25: [1. 1. 1.]

            Q_action = self.get_action(state_new)

            inputs[i] = state

            if done:
                if action >= self.action_space:
                    print("ERROR: action greater than action space\nAction:",action)
                    targets[i][action % self.action_space] = reward
                else:
                    targets[i][action] = reward
            else:
                #print(i, action, Q_action, self.action_space)
                #TODO: verify the cause of action being greater than action_space
                if action >= self.action_space:
                    print("ERROR: action greater than action space\nAction:",action)
                    targets[i][action % self.action_space] = reward + self.gamma * np.amax(Q_action) - targets[i][action]
                else:
                    targets[i][action] = reward + self.gamma * np.max(Q_action) - targets[i][action]

        #add another dimension to state observation (shape[0] <-- self.batch_size)
        #inputs = np.stack((inputs,), axis=1)
        #inputs = np.expand_dims(inputs, axis=1)


        #train network to output Q function
        self.model.fit(inputs, targets, epochs=1,
                       batch_size=self.batch_size,
                       verbose=0)
        #Verbosity mode. 0 = silent, 1 = progress bar,
        #                2 = one line per epoch.

    def display_statisics_to_console(self):
        length = self.memory.get_capacity()
        scores = self.memory.get_scores()
        print('Average Score:',sum(scores)/length)

        print("Average action choice:")
        actions = self.memory.get_actions()
        for i in range(self.action_space):
            print("-->", actions.count(i)/length)

    def graph_scores(self, num_scores=1000):
        self.memory.graph(num_scores)

    def save_model(self, location):
        _id = str(sum(self.memory.get_scores())/self.memory.get_capacity())[:10]
        now = datetime.datetime.now().strftime("%b-%d_%H:%M_weights_avg_score~")
        self.model.save_weights(location + now + _id + '.h5')

    def load_model(self, name):
        self.model.load_weights(name)


'''
if __name__ == "__main__":
    agent = DQN()
'''
