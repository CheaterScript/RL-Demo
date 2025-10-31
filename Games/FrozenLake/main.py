import gymnasium as gym
import numpy as np
import random as rd
from tqdm import trange

env = gym.make("FrozenLake-v1", map_name="4x4",
               is_slippery=False, render_mode="rgb_array")
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space", env.observation_space)
print("Sample observation", env.observation_space.sample())
print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())


def create_Qtable(num_actions, num_observation):
    Qtable = np.zeros((num_observation, num_actions))
    print("Qtable size:", Qtable.shape, num_actions, num_observation)
    return Qtable


Qtable_frozenlake = create_Qtable(env.action_space.n, env.observation_space.n)


def greedy_policy(Qtable, state):
    action = np.argmax(Qtable[state][:])
    return action


def epsilon_greedy_policy(Qtable, state, epsilon):
    if rd.uniform(0, 1) <= epsilon:
        action = env.action_space.sample()
    else:
        action = greedy_policy(Qtable, state)
    return action


learning_rate = 0.7
gamma = 0.99


def train(episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in trange(episodes):
        epsilon = min_epsilon + \
            (max_epsilon - min_epsilon) * np.exp(-decay_rate *
                                                 episode)  # np.exp(-decay_rate * episode)
        state, info = env.reset()
        step = 0

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)

            Qtable[state][action] += learning_rate * \
                (reward + gamma *
                np.max(Qtable[new_state]) - Qtable[state][action])
            
            if terminated or truncated:
                break
            
            state = new_state
    return Qtable
        
