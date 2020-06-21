import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

# import self as self
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
            act = random.randrange(self.num_actions)
            return torch.tensor([act]).to(self.device)
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
        # self.env = gym.make("Pong-v0").unwrapped
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
        return self.env.action_space.n

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
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        # pass the array as a tensor
        screen = torch.from_numpy(screen)
        resize = T.Compose([T.ToPILImage(), T.Resize((40, 90)), T.ToTensor()])
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


######################## PLOTTING ########################


# period: represents the period over which we want to calculate an avg
# values: values represent a list of values representing the number of
#         rewards that the agent got
def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    # length of the dataset must be longer than that of the required period
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1) \
            .flatten(start_dim=0)
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


# plot the duration of each episode and the moving average in 100 episodes
# note that the target reward is 195 over 100 consecutive episodes.
def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title("Training in progress")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    plt.pause(0.001)
    print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:",
          moving_avg[-1])
    if is_ipython: display.clear_output(wait=True)


######################## Training ########################


batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10  # update the target network every 10 episodes
memory_size = 100000
learning_rate = 0.001
num_episodes = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
envManager = CartpoleEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, envManager.num_actions_available(), device)
memory = ReplayMemory(memory_size)

# input dimension
policy_net = DQN(envManager.get_screen_height(), envManager.get_screen_width()).to(device)
target_net = DQN(envManager.get_screen_height(), envManager.get_screen_width()).to(device)
# clone the policy net to target net
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # indicating that the target networks is not in training mode => eval mode
optimizer = optim.Adam(params=policy_net.parameters(), lr=learning_rate)


# this class deals with the second pass to get the predicted q value
class QValues():
    # since we won't create an instance of this class, we won't be using the device
    # that we have already created outside of this class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # static method allows us to use the tagged functions without creating a class first
    @staticmethod
    def get_current(policy_net, states, actions):
        # returns predicted q values from the policy net for the state action pair
        print(states.size())
        print(actions.size())
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    # do we have any final states in our next_state tensor?
    # if we do, we don't pass them to the target net because their associated values are 0
    @staticmethod
    def get_next(target_net, next_states):
        # for each next state, we want to obtain the max q value predicted by the target net
        # final states are represented by all pixels being 0
        # we are filtering through all the locations to get where all pixels are 0
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0]. \
            eq(0).type(torch.bool)  # if max is 0, we know it's final state
        # we don't want to pass these final state locations to target net
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        # how many next states are in the next_states tensor?
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        # set values at non_final_state_locations to be the same as
        # maximum predicted q values from the target net across each action
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        # values will be such that:
        # 0's as the q-values that are the final states, and
        # for each non-final state, the tensor values contains the max predicted
        # q-values across all actions for each non-final state
        return values


def extract_tensors(experiences):
    # transpose the input experiences into a batch of experiences
    # i.e. transform:
    # [Experience(1, 1, 1, 1), Experience(2, 2, 2, 2)] into:
    # Experience((1, 2), (1, 2), (1, 2), (1, 2))
    batch = Experience(*zip(*experiences))
    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)
    return (t1, t2, t3, t4)


# values in the list are the number of rewards that the agent receives
# 1 reward per move to keep the game going => 1 reward = 1 time step of duration
episode_durations = []  # list of values that would be used later


def play_game():
    for episode in range(num_episodes):
        envManager.reset()
        state = envManager.get_state()

        for timestep in count():
            # agent uses the policy net to explore or exploit
            action = agent.select_action(state, policy_net)
            reward = envManager.take_action(action)
            next_state = envManager.get_state()
            memory.push(Experience(state, action, next_state, reward))
            state = next_state

            # training
            if memory.can_return_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()  # set the gradients of the weights and biases to 0
                loss.backward()  # back propogation after zero_grad to avoid accumulation of grad
                optimizer.step()  # updates weights and biases from back prop calculation

            if envManager.done:
                episode_durations.append(timestep)
                plot(episode_durations, 100)
                break

        if episode % target_update == 0:
            # update the target net weights with the policy net's weights
            target_net.load_state_dict(policy_net.state_dict())

    envManager.close()


def show_unprocessed_screen():
    screen = envManager.render("rgb_array")
    plt.figure()
    plt.imshow(screen)
    plt.title("Non-processed screen example")
    plt.show()


def show_processed_screen():
    screen = envManager.get_processed_screen()
    plt.figure()
    plt.imshow(screen.squeeze(0).permute(1, 2, 0), interpolation="none")
    plt.title("Processed screen example")
    plt.show()


play_game()
