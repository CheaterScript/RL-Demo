import gymnasium as gym
import numpy as np
import random as rd
from tqdm import trange

IS_TRAIN = True

env = gym.make("Taxi-v3", render_mode=None if IS_TRAIN else "human")
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


def epsilon_greedy_policy(Qtable, state, epsilon, info):
    if rd.uniform(0, 1) <= epsilon:
        action = env.action_space.sample(info["action_mask"])
    else:
        action = greedy_policy(Qtable, state)
    return action


def train(episodes, min_epsilon, max_epsilon, decay_rate, env, max_steps, Qtable):
    for episode in trange(episodes):
        epsilon = min_epsilon + \
            (max_epsilon - min_epsilon) * np.exp(-decay_rate *
                                                 episode)  # np.exp(-decay_rate * episode)
        state, info = env.reset()
        step = 0

        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon, info)
            new_state, reward, terminated, truncated, info = env.step(action)

            Qtable[state][action] += learning_rate * \
                (reward + gamma *
                 np.max(Qtable[new_state]) - Qtable[state][action])

            if terminated or truncated:
                break

            state = new_state
    return Qtable


def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in trange(n_eval_episodes):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # env.render()
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward
            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


learning_rate = 0.7
gamma = 0.99
max_steps = 99
n_eval_episodes = 25000
eval_seed = []

if IS_TRAIN:
    Qtable_frozenlake = train(n_eval_episodes, 0.05,
                              1, 0.0005, env, max_steps, Qtable_frozenlake)
    np.save('array_file.npy', Qtable_frozenlake)
else:
    Qtable_frozenlake = np.load('array_file.npy')

# Evaluate our Agent
mean_reward, std_reward = evaluate_agent(
    env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
