# This is a simpler version of CartPole.py
# http://gym.openai.com/docs/#observations

import gym
# env = gym.make("CartPole-v0")
env = gym.make("Pong-v0")
env.reset()


def play_game_random():
    for _ in range(500):
        env.render()
        env.step(env.action_space.sample())
    env.close()


def play_game_with_observation():
    for episode in range(10):
        observation = env.reset()   # observation stores the environment
        for step in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()  # still, we are choosing a random action
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} steps".format(step + 1))
                break
    env.close()


play_game_with_observation()
