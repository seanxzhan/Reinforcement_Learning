import gym
# env = gym.make("CartPole-v0")
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())
# env.close()
import math
import random
import numpy as py
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import self as self
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# for display:
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython: from IPython import display


# deep q-network
# we need a policy network and a target network
# network consists of a 2 fully connected hidden layers and an output layer
# DQN extends nn.Module
class DQN(nn.Module):
    def __init__(self, img_h, img_w):
        super().__init__()

        # no convolutional layers below
        # Linear layers are connected layers
        # 3 in the following line means there are three primary colors RGB
        # first linear layer has 24 outputs
        self.fc1 = nn.Linear(in_features=img_h * img_w * 3, out_features=24)
        # second linear layer has 32 outputs
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        # number of outputs is 2 because the cart will either move left or right
        self.out = nn.Linear(in_features=32, out_features=2)

    # required to implement a forward method of the nn.Module class
    # implement a forward pass to the network
    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


# agent's experience
Experience = namedtuple("Experience",
                        ("state", "action", "next_state", "reward"))


# Replay Memory
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0  # how many experiences we have added to memory

    # push stores experiences and replays memory
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # get rid of the oldest experience and add the most recent one
            # manually update exp at an index CAN BE IMPROVED
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    # return a boolean indicating whether we can return a sample from memory
    def can_return_sample(self, batch_size):
        return len(self.memory) >= batch_size

    # return a random sample of the agent's experiences
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


# Use epsilon greedy strategy to choose between exploration v. exploitation
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    # epsilon decays so that we lean towards exploitation as we explore
    def get_explore_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1. * current_step * self.decay)


# A.I.
class Agent():
    # strategy will use an instance of the EpsilonGreedyStrategy
    # note that there are only 2 number of actions: left, right
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    # a function to decide whether the agent will explore or exploit
    def select_action(self, state, policy_net):
        rate = self.strategy.get_explore_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():  # explore
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)
        else:  # exploit
            # passing data to policy netword without gradient tracking
            # (this model is not for training yet)
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)


class CartpoleEnvManager():
    def __init__(self, device):
        self.device = device
        # unwrap allows behind-scene access
        self.env = gym.make("CartPole-v0").unwrapped
        self.env.reset()
        self.current_screen = None  # at the start of an episode, no screen yet
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode)

    def num_actions_available(self):
        return not self.env.action_space.n

    def take_action(self, action):
        # action.item(): action that would be passed should be a tensor
        # item returns the value of the tensor
        _, reward, self.done, _ = self.env.step(action.item())
        # returns reward wrapped in tensor
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def crop_screen(self, screen):
        h = screen.shape[1]
        top = int(h * 0.4)
        bottom = int(h * 0.8)
        screen = screen[:, top:bottom, :]
        return screen

    def get_processed_screen(self):
        # transpose the matrix into arrays by height by width
        screen = self.render("rgb_array").transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def transform_screen_data(self, screen):
        # store screen into an array
        screen = py.np.ascontiguousarray(screen, dtype=py.np.float32) / 255
        # pass the array as a tensor
        screen = torch.from_numpy(screen)
        resize = T.Compose([T.ToPILImage(), T.Resize((40, 90)), T.ToTensor])
        return resize(screen).unsqueeze(0).to(self.device)

    # return the current state of env as a processed image
    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1  # to represent a single state

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
envManager = CartpoleEnvManager(device)
envManager.reset()
screen = envManager.render("rgb_array")

plt.figure()
plt.imshow(screen)
plt.title("Non-processed screen example")
plt.show()