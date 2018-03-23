 ###########################################################
#   Mini Web of Bits Virtual Agent using Q-Leaning Policy   #
#                                                           #
#              Developed by Nathan Shepherd                 #
#                                                           #
 ###########################################################
# Source Code: https://github.com/nathanShepherd/Intelligent-Interface

import gym
import universe
import numpy as np

import time
import random
from CustomDQN_March_18 import DQN

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

  click = 1
  action = coord_to_event(xcoord, ycoord, click)
  print("action: ",action)
  
  #return list of vnc events
  return action

''' Waits for three games until game properly initializes '''
''' TODO: Analyze instructions to pick apropriate Agent   '''
def get_obs_space(env):
  observation = env.reset()
  done = False

  while not done:
    #agent takes an action for each observation
    action_n = [observe_and_take_random_action(obs) for obs in observation]
    observation, reward_n, done_n, info = env.step(action_n)
    
    #print("%%%%%\nInfo:", info['n'][0])
    env_id = info['n'][0]['env_status.episode_id']

    if env_id != None and int(env_id) > 2:
      print("Observation space intialized and returning shape as input to model")
      print("\nObservation: ",observation)
      done = True

  x = observation[0]['vision']
  #crop observation window to fit mwob window
  crop = np.array(x[75:75+210, 10:10+160, :])

  return crop.shape

def reset_mouse_pos(x, y):
  x = 90
  y = 170
  return x, y

def update_position(move, x, y, velocity):
  x_min = 10; x_max = 170
  y_min = 125; y_max = 280
  
  penalty = 0
  #move left, right; check bounds
  if move == 0: x -= velocity
  if move == 1: x += velocity

  #move up, down; check bounds
  if move == 2: y += velocity
  if move == 3: y -= velocity

  if x > x_max:
    penalty = 0.1
    x = x_max - velocity
  if x <= x_min:
    penalty = 0.1
    x = x_min + velocity
  if y > y_max:
    penalty += 0.1
    y = y_max - velocity
  if y <= y_min:
    penalty += 0.1
    y = y_min + velocity

  click = 0
  if move == 4:  click = 1

  return x, y, click, penalty

def get_training_data(env):
  state = env.reset()

  training_data = []
  for episode in range(initial_games):
    game_memory = []
    
    for frame  in range(goal_steps):
      #agent takes an action for each observation
      action_n = [observe_and_take_random_action(obs) for obs in state]
      state_next, reward_n, done_n, info = env.step(action_n)
      print("\n\n~~~~~~~~ Reward: ", reward_n, "\n\n~~~~~~~~")
      print("\n\n~~~~~~~~ Action: ", action_n, "\n\n~~~~~~~~")
      env.render()

      transition = [state, action_n, reward_n, state_next, done_n]
      game_memory.append(transition)

      state = state_next
      ##IS THIS LOOP EVER GOING TO END???
      # TODO: end this loop, observe done_n vector
      if done_n: break

    training_data.append(game_memory)
  return training_data
  

def create_random_samples(init_obs, env, mouse_x_pos, mouse_y_pos, vel):
    # [state, action, reward, state_new, done]
    training_data = []
    
    # just the scores that met threshold:
    accepted_scores = []

    init_obs = env.reset()
    mouse_x_pos, mouse_y_pos = reset_mouse_pos(mouse_x_pos, mouse_y_pos)

    x_min = 10; x_max = 170
    y_min = 125; y_max = 280
    
    for game in range(initial_games):
        if game % 10 == 0:
            print('=========\nObserving random samples: {}%'.format(game*100/initial_games))
            
        score = 0
        penalty = 0
        game_memory = []
        state = init_obs
        #mouse_x_pos, mouse_y_pos = reset_mouse_pos(mouse_x_pos, mouse_y_pos)

        #init_obs = env.reset()#????????? may need removal

        for frame in range(goal_steps):
            
            '''
            action = []
            for _ in state:
              #TODO: Take actions across continuous 2D space
              #move = random.randrange(0, action_space)
              move = [random.randrange(x_min, x_max), random.randrange(y_min, y_max)]
              print("Q-Action:", move)
              
              #mouse_x_pos, mouse_y_pos, click, penalty = update_position(move, mouse_x_pos, mouse_y_pos, vel)
              #update = coord_to_event(mouse_x_pos, mouse_y_pos, click)
              click = 1
              penalty = 0
              update = coord_to_event(move[0], move[1], click)
              action.append(update)

              #TODO: ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              #TODO:  ^^^^^^^Figure out why create_random_samples does not^^^^^^^^^^
              #TODO: ^^^^^^move mouse but play random games is working fine^^^^^^^^^^
            '''
            action = [observe_and_take_random_action(obs) for obs in state]

            #print("ACTION::::", action, type(action))
            state_new, reward, done, info = env.step(action)

            reward[0] -= penalty
            #game_memory.append([state, move, reward, state_new, done])
            print("\n\n~~~~~~~~ Reward: ", reward, "\n\n~~~~~~~~")
            print("\n\n~~~~~~~~ Action: ", action, "\n\n~~~~~~~~")
            score += reward[0]

            state = state_new

            env.render()
            
            if done: break

        

        # IF our score >= threshold, we'd like to save [action, obs] pairs
        if score >= score_requirement:
            accepted_scores.append(score)
            
            for data in game_memory:
                training_data.append(data)
                
        init_obs = env.reset()

    training_data_save = np.array(training_data)
    #np.save('lunar_lander_training_data.npy',training_data_save)
    
    print('Average accepted score:',np.mean(accepted_scores))
    print('Median score for accepted scores:',np.median(accepted_scores))
    print("Number of acccepted scores:", len(accepted_scores))
    
    return np.array(training_data)

def get_CustomDQN_Agent(env, action_shape, observation_space):
  print('\n\n}{}{}}{}{}{}{}{}{}{}{}{}{}{')
  #print(env.action_space.keys)# get all possible key events
  #print(env.action_space.buttonmasks)# get "buttonmasks"???
  # checkout universe/universe/spaces/vnc_action_space.py
  # ^^^This repo provides more information about key events


'''Take in coordinates on the screen, return action as a list of VNC events'''
def coord_to_event(x, y, click):
  # Line 1: Move to x, y
  # Line 2 & 3: Click at (x, y)
  # TODO: make actions modular with respect to the task required
  #       consider using the env name to decide which class of
  #       set of actions would make the most sense for the context
  action = [universe.spaces.PointerEvent(x, y, 0),
            universe.spaces.PointerEvent(x, y, click),
            universe.spaces.PointerEvent(x, y, 0)]
  return action

def random_game_loop(env):
  observation = env.reset()

  start_time = time.time()
  for episode in range(10):
    for frame  in range(100):
      #agent takes an action for each observation
      action_n = [observe_and_take_random_action(obs) for obs in observation]
      observation, reward_n, done_n, info = env.step(action_n)
      print('####\nReward:', reward_n,'\n')
      print("####\nInfo:", info, '\n')
      #env.render()

      if time.time() - start_time > 25:#30
        break

#~~~~~~~~~~~[  MAIN  ]~~~~~~~~~~~#
    
#initialize game environment
env = gym.make('wob.mini.ClickButton-v0')
goal_steps = 100#just barely starts at 100,000
score_requirement = -200#-150
initial_games = 500#
num_training_games = 100#>1000

# (left, right, up, down, click) for Click games
action_space = 5
mouse_x_pos = 90
mouse_y_pos = 170
velocity = 30

x_min = 10; x_max = 170
y_min = 125; y_max = 280
 
#TODO: feed instructions through vector space model and train LSTM/CNN/NN
  
if __name__ == "__main__":
  # automatically creates a local docker container
  env.configure(remotes=1, fps=5, vnc_driver='go',
              vnc_kwargs={'encoding':'tight', 'compress_level': 0,
                          'fine_quality_level': 100, 'subsample_level': 0})

  random_game_loop(env)
  
  obs_space= get_obs_space(env)
  print("%%%%%\nThe observation space is:",obs_space,"\n%%%%%")
  print("%%%%%\nThe action space is: continuous 2D\n%%%%%")

  training_data = get_training_data(env)
  
'''
  initial_observation = env.reset()
  training_data = create_random_samples(initial_observation, env,
                                        mouse_x_pos, mouse_y_pos,
                                        velocity)
  
  Agent = DQN(batch_size=64,
              memory_size=50000,
              learning_rate=0.005,
              random_action_decay=0.5,)

  print("Storing training data")
  for datum in training_data:
      s, a, r, s_, done = datum
      Agent.store_transition(s, a, r, s_, done)

  Agent.init_model(obs_space, action_space)
  Agent.train()
'''


















































