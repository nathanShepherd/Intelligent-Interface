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

def reset_mouse_pos():
  mouse_x_pos = 90
  mouse_y_pos = 170

def move_mouse_and_check_boundries(move):
  penalty = 0
  #move left, right; check bounds
  if move == 0: mouse_x_pos -= velocity
  if move == 1: mouse_x_pos += velocity
  if mouse_x_pos > x_max:
    penalty = 0.1
    mouse_x_pos = x_max - velocity
  if mouse_x_pos <= x_min:
    penalty = 0.1
    mouse_x_pos = x_min + velocity

  #move up, down; check bounds
  if move == 2: mouse_y_pos += velocity
  if move == 3: mouse_y_pos -= velocity
  if mouse_y_pos > y_max:
    penalty += 0.1
    mouse_y_pos = y_max - velocity
  if mouse_y_pos <= y_min:
    penalty += 0.1
    mouse_y_pos = y_min + velocity

  click = 0
  if move == 4:  click = 1

  return click, penalty

def create_random_samples(init_obs, env):
    # [state, action, reward, state_new, done]
    training_data = []
    
    # just the scores that met threshold:
    accepted_scores = []
    
    for i in range(initial_games):
        if i % 10 == 0:
            print('Observing random samples: {}%'.format(i*100/initial_games))
            
        score = 0
        game_memory = []
        state = init_obs
        reset_mouse_pos()

        for _ in range(goal_steps):
            #TODO: Take actions across continuous 2D plane
            move = random.randrange(0, action_space)
            print("Q-Action:", q_act)
            
            click, penalty = move_mouse_and_check_boundries(move)
       
            action = coord_to_event(mouse_x_pos, mouse_y_pos, click)

            #print("ACTION::::", action, type(action))
            state_new, reward, done, info = env.step(action)
                
            game_memory.append([state, move, reward - penalty, state_new, done])
            score += reward
            
            if done: break

        # IF our score >= threshold, we'd like to save [action, obs] pairs
        if score >= score_requirement:
            accepted_scores.append(score)
            
            for data in game_memory:
                training_data.append(data)
                
        init_obs = env.reset()

    training_data_save = np.array(training_data)
    #np.save('lunar_lander_training_data.npy',training_data_save)
    
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print("Number of acccepted scores:", len(accepted_scores))
    
    return np.array(training_data)

def get_CustomDQN_Agent(env, action_shape, observation_space):
  print('\n\n}{}{}}{}{}{}{}{}{}{}{}{}{}{')
  #print(env.action_space.keys)# get all possible key events
  #print(env.action_space.buttonmasks)# get "buttonmasks"???
  # checkout universe/universe/spaces/vnc_action_space.py
  # ^^^This repo provides more information about key events


'''Take in coordinates on the screen, return action as a list of VNC events'''
def coord_to_event(xcoord, ycoord, click):
  # Line 1: Move to x, y
  # Line 2 & 3: Click at (x, y)
  # TODO: make actions modular with respect to the task required
  #       consider using the env name to decide which class of
  #       set of actions would make the most sense for the context
  action = [universe.spaces.PointerEvent(xcoord, ycoord, 0),
            universe.spaces.PointerEvent(xcoord, ycoord, click),
            universe.spaces.PointerEvent(xcoord, ycoord, 0)]
  return action

def random_game_loop(env):
  observation = env.reset()

  while True:
    #agent takes an action for each observation
    action_n = [observe_and_take_random_action(obs) for obs in observation]
    observation, reward_n, done_n, info = env.step(action_n)
    print('####\nReward:', reward_n,'\n')
    print("####\nInfo:", info, '\n')
    env.render()

#~~~~~~~~~~~[  MAIN  ]~~~~~~~~~~~#
    
#initialize game environment
env = gym.make('wob.mini.ClickButton-v0')
goal_steps = 1000#2000
score_requirement = -200#-150
initial_games = 50#10000, 250 adequite
num_training_games = 100#>1000

# (left, right, up, down, click) for Click games
action_space = 5
mouse_x_pos = 90
mouse_y_pos = 170
velocity = 5

x_min = 10; x_max = 170
y_min = 125; y_max = 280
 
#TODO: feed instructions through vector space model and train LSTM/CNN/NN
  
if __name__ == "__main__":
  # automatically creates a local docker container
  env.configure(remotes=1, fps=5, vnc_driver='go',
              vnc_kwargs={'encoding':'tight', 'compress_level': 0,
                          'fine_quality_level': 100, 'subsample_level': 0})

  
  obs_space= get_obs_space(env)
  print("%%%%%\nThe observation space is:",obs_space,"\n%%%%%")
  print("%%%%%\nThe action space is: continuous 2D\n%%%%%")

  initial_observation = env.reset()
  training_data = create_random_samples(initial_observation, env)
  
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



















































