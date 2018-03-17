import gym
#import universe

import random
import numpy as np

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.container = []

    def get_sample(self, size):
        return random.sample(self.container, size)
    
    def push_sample(self,state, action, reward, state_next, done):
        self.container.append([state, action, reward, state_next, done])
    
class DQN:
    def __init__(
            self,
            action_space=None,
            observation_space=None,
            learning_rate=0.005,
            reward_decay=0.9,
            random_action_chance=0.9,
            #replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            hidden=None,
            output_graph=False,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.LR = learning_rate
        self.reward_decay = 0.9
        self.epsilon = random_action_chance
        #self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment

        self.hidden = None
        self.output_graph = None

        self.gamma = 0.9
        self.memory = Memory(memory_size)

    def init_model(self, observation_space, nb_actions):#TODO: parameterize
        self.action_space = nb_actions
        self.observation_space = observation_space
        
        model = Sequential()
        model.add(Dense(20, input_shape=(1,) + observation_space))
        model.add(Flatten())
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer='adam',
                      metrics=['accuracy'])
        print(model.summary())

        self.model = model

    def _take_random_action(self):
        return random.randint(0, self.action_space)
    
    def store_transition(self, state, action, reward, state_next, done):
        self.memory.push_sample(state, action, reward, state_next, done)

    def get_action(self, state):
        # decrement epsilon via: e = e_min + (e_max - e_min)^(lambda*time)
        # where lambda is the decay factor
        if random.random() < self.epsilon:
            return self._take_random_action()
        else:
            #add another dimension to state (i.e. batch size is one)
            #state = np.array(state)
            state = np.expand_dims(state, axis=0)
            state = np.stack((state,), axis=1)
            print("State shape:",state.shape)
            print("Predicting action from state")
            return np.argmax(self.model.predict(state))

    def siraj_train(self):
        transitions = self.memory.get_sample(self.batch_size)

        inputs_shape = (self.batch_size,) + self.observation_space
        inputs = np.zeros(inputs_shape)
        targets = np.zeros((self.batch_size, self.action_space))

        for i in range(len(transitions)):
            print("Making target: {} of {}".format(i, len(transitions)))
            trans = transitions[i]
            state = trans[0];
            action = trans[1];
            reward = trans[2];
            state_new = trans[3]
            done = trans[4]

            targets[i] = self.get_action(state)
            #TODO: Consult siraj's code and determine correct shape of action!
            print("Target {}: {}".format(i, targets[i]))
            Q_sa = self.get_action(state_new)

            #add another dimension to state (i.e. batch size is one)
            
            inputs[i] = state

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.max(Q_sa)
        inputs = np.expand_dims(inputs, axis=1)
        #inputs = np.stack((inputs,), axis=1)
        #train network to output Q function
        self.model.train_on_batch(inputs, targets)
        

    def train(self):
        #get training samples from memory
        transitions = self.memory.get_sample(self.batch_size)
        
        states = np.array([ trans[0] for trans in transitions ])
        states_ = np.array([ trans[3] for trans in transitions ])

        pred = self.model.predict(states)
        pred_ = self.model.predict(states_)

        x = np.zeros((self.batch_size, self.observation_space))
        y = np.zeros((self.batch_size, self.action_space))

        for i in range(len(transitions)):
            trans = transitions[i]
            state = trans[0];
            action = trans[1];
            reward = trans[2];
            state_new = trans[3]
            done = trans[4]
            
            target = pred[i]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.amax(pred_[i])

            x[i] = state
            y[i] = target
        
        self.model.fit(x, y, batch_size=self.batch_size,
                             nb_epoch=1, verbose=verbose) 
        
'''
if __name__ == "__main__":
    agent = DQN()
'''















