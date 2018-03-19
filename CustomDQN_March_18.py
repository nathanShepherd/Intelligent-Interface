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

    def push_sample(self,state, action, reward, state_next, done):
        self.container.append([state, action, reward, state_next, done])

    def get_sample(self, size):
        return random.sample(self.container, size)

    def get_scores(self):
        return [datum[2] for datum in self.container]
    
    def get_actions(self):
        return [datum[1] for datum in self.container]

    def get_capacity(self):
        return self.capacity
        
    
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
        return random.randrange(0, self.action_space)
    
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
                    targets[i][action % self.action_space] = reward + self.gamma * np.amax(Q_action)
                else:
                    targets[i][action] = reward + self.gamma * np.max(Q_action)
                
        #add another dimension to state observation (shape[0] <-- self.batch_size)
        #inputs = np.stack((inputs,), axis=1)
        inputs = np.expand_dims(inputs, axis=1)
        
        
        #train network to output Q function
        self.model.fit(inputs, targets, nb_epoch=1, batch_size=self.batch_size, verbose=0)
        #Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        
    def display_statisics_to_console(self):
        length = self.memory.get_capacity()
        scores = self.memory.get_scores()
        print('Average Score:',sum(scores)/length)

        print("Average action choice:")
        actions = self.memory.get_actions()
        for i in range(self.action_space):
            print("-->", actions.count(i)/length)

        
        

        
        
'''
if __name__ == "__main__":
    agent = DQN()
'''

































