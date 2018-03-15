import gym
import universe
import numpy as np

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent


def observe_and_take_random_action(obs):
  # obs is raw (768,1024,3) uint8 screen .
  # The browser window indents the origin of MiniWob by 75 pixels from top and
  # 10 pixels from the left. The first 50 pixels along height are the query.
  if obs is None: return []

  text = obs['text']
  if len(text) > 0:
    print('\n\n}{}{}}{}{}{}{}{}{}{}{}{}{}{')
    print(text)
    print('\n\n}{}{}}{}{}{}{}{}{}{}{}{}{}{')

  x = obs['vision']
  crop = np.array(x[75:75+210, 10:10+160, :]) #mwob coordinates crop
  # crop tensor shape --> (210, 160, 3)
  # crop becomes   [(3, x, 2) matrix, 
  #  a tensor:      (3, y, 3) matrix (value 0-255), 
  #                 (3, y, 3) matrix (value 0-255)]
  
 
  xcoord = np.random.randint(0, 160) + 10# add intelligence
  ycoord = np.random.randint(0, 160) + 75 + 50# add intelligence

  # Line 1: Move to x, y
  # Line 2 & 3: Click at (x, y)
  action = [universe.spaces.PointerEvent(xcoord, ycoord, 0),
            universe.spaces.PointerEvent(xcoord, ycoord, 1),
            universe.spaces.PointerEvent(xcoord, ycoord, 0)]
  
  #return list of vnc events
  return action

def use_keral_rl(env, action_shape):
  print('\n\n}{}{}}{}{}{}{}{}{}{}{}{}{}{')
  #print(env.action_space.keys)# get all possible key events
  #print(env.action_space.buttonmasks)# get "buttonmasks"???
  # checkout universe/universe/spaces/vnc_action_space.py
  # ^^^This repo provides more information about key events
  nb_actions = action_shape#TODO: get actual shape of action_space

  # Next, we build a very simple model.
  model = Sequential()
  model.add(Flatten(input_shape=(1,) + env.observation_space.shape))#TODO: get observation_space
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(16))
  model.add(Activation('relu'))
  model.add(Dense(nb_actions))
  model.add(Activation('linear'))
  print(model.summary())

  # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
  # even the metrics!
  memory = SequentialMemory(limit=50000, window_length=1)
  policy = BoltzmannQPolicy()
  dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
  dqn.compile(Adam(lr=1e-3), metrics=['mae'])

  # Okay, now it's time to learn something! We visualize the training here for show, but this
  # slows down training. You can always safely abort the training prematurely using Ctr + C
  # TODO: examine keras-rl/examples in order to crop obervation space during input to DQN
  dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

  # After training is done, we save the final weights.
  dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

  # Finally, evaluate our algorithm for 5 episodes.
  dqn.test(env, nb_episodes=5, visualize=True)


def random_game_loop():
  while True:
    #agent takes an action for each observation
    action_n = [observe_and_take_random_action(obs) for obs in observation_n]
    observation_n, reward_n, done_n, info = env.step(action_n)
    env.render()
  


#initialize game environment
env = gym.make('wob.mini.ClickButton-v0')

# automatically creates a local docker container
env.configure(remotes=1, fps=5, vnc_driver='go',
              vnc_kwargs={'encoding':'tight', 'compress_level': 0,
                          'fine_quality_level': 100, 'subsample_level': 0})
observation_n = env.reset()

use_keral_rl(env, 12)
