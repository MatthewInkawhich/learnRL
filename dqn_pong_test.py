from __future__ import print_function
from __future__ import division
from collections import namedtuple
import collections
from itertools import count
from PIL import Image
import gym
import os
import math
import random
import cv2
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

##################################################################
### Initialize Enviroment
##################################################################
PATH = "checkpoint/test3.pt"
# Initialize gym environment
env = gym.make('Pong-v0')
# print(env.action_space)
# print(env.unwrapped.get_action_meanings())
NUM_ACTIONS = 3

# Pong-specific action mapping. Must map model output argmax {0, 1, 2} to the action
# indexes used by the environment
# Model output  |   atari action    |   action
#---------------|-------------------|---------------
#       0       |          0        |   NO-OP
#       1       |          2        |   Move UP
#       2       |          3        |   Move DOWN
action_translator = {0:0, 1:2, 2:3}

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



##################################################################
### Define Model
##################################################################
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(in_features=32*9*9 , out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    # return Q values of each action



##################################################################
### Frame processing
##################################################################
# FrameBuffer is a FIFO buffer of fixed length "size". This structure is
# designed to store the latest n frames that we have observed, as this is what
# we stack and feed to the model for temporal information.
class FrameBuffer:
    def __init__(self, size):
        self.empty_fill = None
        self.size = size
        self.buffer = collections.deque(self.size*[self.empty_fill], self.size)

    def push(self, frame):
        self.buffer.append(frame)

    def is_full(self):
        for item in self.buffer:
            if item is None:
                return False
        return True

    def get_frame_tensor(self):
        assert(self.is_full()), "Attempted to get_frames when buffer is not full"
        return torch.stack(list(self.buffer)).unsqueeze_(0)

    def plot(self):
        assert(self.is_full()), "Attempted to get_frames when buffer is not full"
        frames = torch.stack(list(self.buffer))
        f, axarr = plt.subplots(1, 4)
        axarr[0].axis('off')
        axarr[0].imshow(frames[0], cmap='gray')
        axarr[1].axis('off')
        axarr[1].imshow(frames[1], cmap='gray')
        axarr[2].axis('off')
        axarr[2].imshow(frames[2], cmap='gray')
        axarr[3].axis('off')
        axarr[3].imshow(frames[3], cmap='gray')
        plt.show()


# This function reads the current atari screen into a numpy array, converts
# from BGR to grayscale, and resizes it to 84x84. A torch.tensor object is
# then returned.
def process_frame():
    screen = env.render(mode='rgb_array')
    gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (84, 110))[18:102,:]
    return torch.tensor(gray, dtype=torch.float32) / 255



##################################################################
### Training Setup
##################################################################
EPSILON = 0.1

policy_net = DQN().to(device)
policy_net.load_state_dict(torch.load(PATH, map_location='cpu')['policy_net_sd'])
policy_net.eval()


# Select an action with epsilon greedy exploration policy.
def select_action(state):
    global EPSILON
    sample = random.random()
    if (sample > EPSILON):
        # Choose greedy action
        with torch.no_grad():
            ans = policy_net(state).max(1)[1].view(1, 1)
    else:
        # Choose random action
        ans = torch.tensor([[random.randrange(NUM_ACTIONS)]], device=device, dtype=torch.long)
    return ans



##################################################################
### Training Loop
##################################################################
class Stats:
    def __init__(self):
        self.ready = False
        self.best_success_count = 0
        self.reset()

    def reset(self):
        self.success_count = 0
        self.total_episode_count = 0
        self.total_iter_count = 0
        self.total_reward = 0
        self.total_loss = 0
        self.action_counts = [0] * NUM_ACTIONS


NUM_EPISODES = 10
FRAME_HISTORY_SIZE = 4
stats = Stats()
frame_buffer = FrameBuffer(FRAME_HISTORY_SIZE)

# Run between 4 and 30 random actions at start of each episode to:
#   1) Fill the frame_buffer
#   2) Offset starting frame to reduce overfitting
def run_noop_iters():
    iters = FRAME_HISTORY_SIZE
    for i in range(iters):
        env.step(env.action_space.sample())
        curr_frame = process_frame()
        frame_buffer.push(curr_frame)

# Training loop
for i_episode in range(NUM_EPISODES):
    # Reset env for new episode
    env.reset()
    # Run noop iterations to fill frame_buffer
    run_noop_iters()
    # Get initial state
    state = frame_buffer.get_frame_tensor()

    # Run episode until someone scores or a max MAX_TIMESTEPS limit is hit
    for t in count():
        stats.total_iter_count += 1
        # Select an action
        action = select_action(state)
        #print("action:", action, action.item())
        stats.action_counts[action.item()] += 1
        # Translate [0,2] action space to atari action space {0, 2, 3}
        atari_action = action_translator[action.item()]
        #print("atari action:", atari_action)
        # Perform the action
        _, reward, done, _ = env.step(atari_action)

        env.render()
        sleep(0.025)

        reward = torch.tensor([reward], device=device)

        # Update frame_buffer
        curr_frame = process_frame()
        frame_buffer.push(curr_frame)

        # Set next_state depending on reward
        next_state = frame_buffer.get_frame_tensor()
#        if (reward == 0):
#            next_state = frame_buffer.get_frame_tensor()
#        else:
#            next_state = None
#            if reward > 0:
#                stats.success_count += 1
#            stats.total_episode_count += 1
#            stats.total_reward += reward.item()

        # Update state
        state = next_state

        # If done...
        #if next_state is None:
        if done:
            print("Episode done!! Reward:", reward.item())
            break


# Print stats when done
print("*************************************")
print("Episodes:", NUM_EPISODES)
print("Success: {}/{} = {}".format(stats.success_count, stats.total_episode_count, stats.success_count/stats.total_episode_count))
print("Avg. iters per episode:", stats.total_iter_count/stats.total_episode_count)
print("Total reward:", stats.total_reward)
print("Action counts:")
for i in range(NUM_ACTIONS):
    print("{}:{}".format(i, stats.action_counts[i]))
print("*************************************")


print("Finished!")
