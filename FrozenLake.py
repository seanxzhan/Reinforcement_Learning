# Time complexity: O(n^2). Inefficient.

import numpy as np
import gym
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")

action_space_sz = env.action_space.n
state_space_sz = env.observation_space.n

q_table = np.zeros((state_space_sz, action_space_sz))

# episode will terminate if episode doesn't end in 100 steps
num_episodes = 10000
max_steps_per_episode = 100

# alpha
learning_rate = 0.1
# gamma
discount_rate = 0.99

# exploration v. exploitation trade off
# exploration: go somewhere blindly
# exploitation: go somewhere that will give max q value
exploration_rate = 1  # agent will start out exploring
max_exploration_rate = 1
min_exploration_rate = 0.01
# agent will start to exploit more as it learns more
exploration_decay_rate = 0.001

rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):
        # epsilon value (a random number between 0 and 1):
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            # agent will choose to exploit the env
            action = np.argmax(q_table[state, :])
        else:
            # sample an action randomly
            action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        # update our q_table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * \
                                 (reward + discount_rate * np.max(q_table[new_state, :]))
        state = new_state
        rewards_current_episode += reward
        if done:
            break
    # exploration rate decay:
    exploration_rate = min_exploration_rate + \
                       (max_exploration_rate - min_exploration_rate) * \
                       np.exp(-exploration_decay_rate * episode)
    rewards_all_episodes.append(rewards_current_episode)

rewards_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000
print("*** Average reward per 100 episodes *** \n")
for r in rewards_episodes:
    print(count, ": ", str(sum(r / 1000)))
    count += 1000

# watch agent play the game
for episode in range(3):
    state = env.reset()
    done = False
    print("*** EPISODE ", episode + 1, "*** \n\n\n")
    time.sleep(1)
    for step in range(max_steps_per_episode):
        clear_output(wait=True) # so that screen stays put until there's another output
        env.render()
        time.sleep(0.5)
        action = np.argmax(q_table[state, :])
        new_state, reward, done, info = env.step(action)
        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("Agent Won")
                time.sleep(3)
            else:
                print("Agent fell through a hole")
                time.sleep(3)
            clear_output(wait=True)
            break
        state = new_state
env.close()

