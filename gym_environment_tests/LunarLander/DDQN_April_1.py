import gym
#import universe

#for saving models with date information
import datetime

import random
import numpy as np

import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.container = []

    def push_sample(self,state, action, reward, state_next, done):
        if self.is_full():
            del self.container[random.randint(0, len(self.container) - 1)]
            
        self.container.append([state, action, reward, state_next, done])

    def is_full(self):
        return len(self.container) >= self.capacity

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

            epsilon_min=0.2,
            reward_decay=0.9,
            learning_rate=0.005,
            random_action_chance=0.9,
            random_action_decay=0.99,#depreciated
            
            replace_target_iter=2000,

            batch_size=32,
    ):
        self.action_space = action_space
        self.observation_space = observation_space

        self.memory_size = memory_size

        self.epsilon_min= epsilon_min
        self.gamma = reward_decay
        self.LR = learning_rate
        self.epsilon = random_action_chance
        self.random_action_decay = random_action_decay

        self.replace_t_iter = replace_target_iter

        self.batch_size = batch_size

        self.memory = Memory(memory_size)

        self.counter = 1

    def init_model(self, observation_space, nb_actions, hidden=[50, 50, 100, 25]):
        self.target_net = self.define_model(observation_space, nb_actions, hidden)
        self.eval_net = self.define_model(observation_space, nb_actions, hidden)

    def define_model(self, observation_space, nb_actions, hidden):
        #TODO: Add regularization
        self.action_space = nb_actions
        self.observation_space = observation_space
        
        model = Sequential()
        model.add(Dense(hidden[0], input_shape=(1,) + observation_space))
        model.add(Flatten())

        for i in range(1, len(hidden) - 1):
            model.add(Dense(hidden[i]))#50, 100, 50
            model.add(Activation('relu'))
            
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))

        adam = Adam(lr=self.LR)

        model.compile(loss='mse', optimizer=adam,
                      metrics=['accuracy'])
        print(model.summary())

        return model

    def _take_random_action(self):
        return random.randrange(0, self.action_space)
    
    def store_transition(self, state, action, reward, state_next, done):
        self.memory.push_sample(state, action, reward, state_next, done)

    '''less likely to randomly take action relative to get_action (improved stability)'''
    def use_policy(self, state):        
        state = np.expand_dims(state, axis=0)
        state = np.stack((state,), axis=1)

        pred = self.target_net.predict(state)
        return np.argmax(pred)

    def evaluate_state(self, state):
        state = np.expand_dims(state, axis=0)
        state = np.stack((state,), axis=1)
        print(state)
        pred = self.eval_net.predict(state)
        
        return pred[0]

    def get_target(self, state):
        state = np.expand_dims(state, axis=0)
        state = np.stack((state,), axis=1)
        pred = self.target_net.predict(state)

        return pred[0]
        

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
            pred = self.target_net.predict(state)
            #print("model prediction:", p)
            return np.argmax(pred)

    def train(self):
        #get training samples from memory
        transitions = self.memory.get_sample(self.batch_size)
        
        inputs_shape = (self.batch_size,) + self.observation_space
        inputs = np.zeros(inputs_shape)
        targets = np.zeros((self.batch_size, self.action_space))

        #decrease random action probability
        '''
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.random_action_decay
        '''
        self.epsilon = min([0.9, (1000/self.counter)+0.2])
        self.counter += 1
        if self.counter % self.replace_t_iter == 0:#replace weights of evaluation net
            print("Updating evaluation network")
            self.eval_net.set_weights(self.target_net.get_weights())

        #print("Training on batch of size:", self.batch_size)
        for i in range(len(transitions)):
            #print("Making target: {} of {}".format(i, len(transitions)))
            trans = transitions[i]
            state = trans[0];
            action = trans[1];
            reward = trans[2];
            state_new = trans[3]
            done = trans[4]

            inputs[i] = state

            #predict reward distribution of state from target model
            targets[i] = self.get_target(state)
            #print("\n\nTarget",targets[i])

            est_val_of_state_next = self.evaluate_state(state_new)
            future_action = np.argmax(est_val_of_state_next)
            #print("Reward",reward, end="")
            #print("Est reward next:",est_val_of_state_next)
            #print("reward", reward)
            #print("Action",action)
            
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * est_val_of_state_next[future_action]
                #print("updated target", targets[i])
                
        #add another dimension to state observation (self.batch_size --> shape[0])
        #inputs = np.stack((inputs,), axis=1)
        inputs = np.expand_dims(inputs, axis=1)
        
        #train network to output Q function
        self.target_net.fit(inputs, targets, epochs=1,
                            batch_size=self.batch_size, verbose=0)
        #Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        
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
        #use keras save model
        _id = str(sum(self.memory.get_scores())/self.memory.get_capacity())[:10]
        now = datetime.datetime.now().strftime("%b-%d_avg_score~")
        self.target_net.save(location + now + _id + '.h5')

    def load_model(self, name):
        self.target_net = load_model(name)
        self.eval_net = load_model(name)
        

        
        
'''
if __name__ == "__main__":
    agent = DQN()
'''

































