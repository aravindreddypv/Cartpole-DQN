import gym
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import random
from gym import wrappers
from keras.models import load_model


def getAction(t_observation):
    observation_array = np.array(t_observation)
    observation_array = observation_array.reshape(1, 4)
    tt = model.predict(observation_array)[0]
    # print tt
    return np.argmax(tt)


QEpisodes = 1000
Epsilon = 1
env = gym.make('CartPole-v0')
# env = wrappers.Monitor(env, 'cartpole-experiment-1')
model = load_model('model1.h5')

for i_episodes in range(QEpisodes):
    # Epsilon += (1.4 / QEpisodes)
    print i_episodes, Epsilon
    observation = env.reset()
    print observation
    for t in range(200):
        env.render()
        temp_observation = observation[:]
        action = getAction(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
