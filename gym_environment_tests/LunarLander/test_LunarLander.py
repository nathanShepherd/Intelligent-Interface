import gym
import numpy as np

env = gym.make('LunarLander-v2')

def play_random(viz=True):
    obs = [env.reset()]
    total_reward = 0
    terminal = False

    while not terminal:
        if viz: env.render()
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        obs.append(observation)
        total_reward += reward
        
    return obs

if __name__ == "__main__":
    frames = play_random(True)
    sorts = []
    for i in range(len(frames[0])):
        sorts.append([])
        for j in range(len(frames)):
            sorts[-1].append(frames[j][i])

    aux = []
    for k in range(1000):
        if k%100==0:print(k*100/1000)
        frames = play_random(viz=False)        
        aux.append(frames)
        
    aux = [np.array(m).T for m in aux]
    total = []
    for i in range(len(aux[0])):
        total.append([])
        for j in range(len(aux)):
            total[-1].append( np.mean(aux[j][i]) )

    for i in range(len(total)):
        print(i, np.mean(total[i]))
    
