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
RUN_NAME = "test2"
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
### Replay Memory
##################################################################
# The Transition namedtuple is the object that is stored in replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# The ReplayMemory class provides and container and interface for
# the replay memory buffer. This circular buffer stores transitions, and
# provides a means to sample a batch randomly
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



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
        return torch.stack(list(self.buffer)).unsqueeze_(0).to(device)

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
BATCH_SIZE = 32
DOUBLE_DQN = False
EPSILON = 1.0
GAMMA = 0.999
ANNEAL_TO = 0.1
ANNEAL_OVER = 1000000
ANNEAL_STEP = (EPSILON - ANNEAL_TO) / ANNEAL_OVER
TARGET_UPDATE = 1000

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00005)
memory = ReplayMemory(100000)


# Select an action with epsilon greedy exploration policy.
def select_action(state):
    global EPSILON
    sample = random.random()
    if (EPSILON > ANNEAL_TO):
        EPSILON -= ANNEAL_STEP
    if (sample > EPSILON):
        # Choose greedy action
        with torch.no_grad():
            ans = policy_net(state).max(1)[1].view(1, 1)
    else:
        # Choose random action
        ans = torch.tensor([[random.randrange(NUM_ACTIONS)]], device=device, dtype=torch.long)
    return ans


def optimize_model():
    if len(memory) < BATCH_SIZE:
        print("Memory does not contain enough samples to make a batch!")
        print("len(memory) = " + str(len(memory)) + "\tBATCH_SIZE = " + str(BATCH_SIZE))
        return


    # Sample a batch from replay memory
    transitions = memory.sample(BATCH_SIZE)
    # Break the batch into tuples of size BATCH_SIZE for each transition element
    state_tuple, action_tuple, next_state_tuple, reward_tuple = zip(*transitions)

    # Handle next states
    next_state_list = list(next_state_tuple)
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(list(map(lambda s: s is not None, next_state_list)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in next_state_list if s is not None])

    # Create batch tensors from these tuples
    state_batch = torch.cat(state_tuple)
    action_batch = torch.cat(action_tuple)
    reward_batch = torch.cat(reward_tuple)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # values of the actions that we actually took.
    predicted_q_values = policy_net(state_batch).gather(1, action_batch)

    if DOUBLE_DQN:
        next_state_q_values = torch.zeros(BATCH_SIZE, device=device)
        # Compute argmax Q(s', a) (using policy net) for Double DQN
        # Note that the argmax is stored in .max(1)[1]
        with torch.no_grad():
            next_policy_net_choices = policy_net(non_final_next_states).max(1)[1].detach()
        # Use Q values from the target network that correspond to the action that the policy Q-net chose
        next_state_q_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_policy_net_choices.unsqueeze(1)).squeeze().detach()
    else:
        next_state_q_values = torch.zeros(BATCH_SIZE, device=device)
        # Compute V(s_{t+1}) for all next states. Note that V(s_{t+1}) = max(Q(s',a))
        next_state_q_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the target Q values
    target_q_values = (next_state_q_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(predicted_q_values, target_q_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


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


NUM_EPISODES = 10000000
NUM_WARMSTART = 500
FRAME_HISTORY_SIZE = 4
MAX_NOOP_ITERS = 30
PROGRESS_INTERVAL = 1000
stats = Stats()
frame_buffer = FrameBuffer(FRAME_HISTORY_SIZE)

# Run between 4 and 30 random actions at start of each episode to:
#   1) Fill the frame_buffer
#   2) Offset starting frame to reduce overfitting
def run_noop_iters():
    iters = random.randint(FRAME_HISTORY_SIZE, MAX_NOOP_ITERS)
    for i in range(iters):
        env.step(env.action_space.sample())
        curr_frame = process_frame()
        frame_buffer.push(curr_frame)

# Training loop
for i_episode in range(NUM_WARMSTART + NUM_EPISODES):
    # Reset env for new episode
    env.reset()
    # Run noop iterations
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
        _, reward, _, _ = env.step(atari_action)
        reward = torch.tensor([reward], device=device)

        # Update frame_buffer
        curr_frame = process_frame()
        frame_buffer.push(curr_frame)

        # Set next_state depending on reward
        if (reward == 0):
            next_state = frame_buffer.get_frame_tensor()
        else:
            next_state = None
            if reward > 0:
                stats.success_count += 1
            stats.total_episode_count += 1
            stats.total_reward += reward.item()

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Update state
        state = next_state

        # Perform an optimization step if not in the warmstart stage
        if i_episode >= NUM_WARMSTART:
            #print("done with warmstart stage...")
            #print("len(memory):", len(memory))
            stats.total_loss += optimize_model()
        else:
            # Warmstart iter... clear stats
            stats.reset()

        # If done...
        if next_state is None:
            #print("Episode done!! Reward:", reward.item())
            break

    # Print statistics and save checkpoint if it is best one yet
    if i_episode > NUM_WARMSTART and i_episode % PROGRESS_INTERVAL == 0:
        # Print stats
        if stats.ready:
            print("*************************************")
            print("Episode:", i_episode)
            print("Last {} episodes...".format(PROGRESS_INTERVAL))
            print("Success: {}/{} = {}".format(stats.success_count, stats.total_episode_count, stats.success_count/stats.total_episode_count))
            print("Avg. iters per episode:", stats.total_iter_count/stats.total_episode_count)
            print("Total reward:", stats.total_reward)
            print("Avg. loss:", stats.total_loss/stats.total_iter_count)
            print("Action counts:")
            for i in range(NUM_ACTIONS):
                print("{}:{}".format(i, stats.action_counts[i]))
            print("*************************************")

            # Save checkpoint if it is best one yet
            if stats.success_count > stats.best_success_count:
                print("Saving model...")
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save({"i_episode":i_episode,
                            "policy_net_sd":policy_net.state_dict(),
                            "optimizer_sd":optimizer.state_dict()}, './checkpoint/' + RUN_NAME + '.pt')

        # Reset stats for next progress interval
        stats.reset()
        stats.ready = True


    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


print("Finished!")
