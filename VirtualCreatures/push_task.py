#!/usr/bin/env python3
import random

import numpy as np
import torch

from environments import PushEnv, Action
from agents import ContinuousRPOAgt
from utils import RLMemory

from loguru import logger


OUT_DISP = True
DT = 1 / 30
SIZE = 800
RND_SEED = 42
L_RATE = 3e-4
ANNEAL_LR = False
TOT_TIMESTEPS = 8000000
NUM_MINI_BATCHES = 32
NUM_STEPS = 2048 // 2  # Given that we have two agents in the environment, data will be gathered twice as fast

# Compute task parameters based on given configuration
batch_size = NUM_STEPS
minibatch_size = batch_size // NUM_MINI_BATCHES
num_updates = TOT_TIMESTEPS // batch_size

# Make the environment as repeatable as possible
random.seed(RND_SEED)
np.random.seed(RND_SEED)
torch.manual_seed(RND_SEED)
torch.backends.cudnn.deterministic = True

# Instantiate a push environment
env = PushEnv(SIZE, DT, out=OUT_DISP)

try:
    # Instantiate the agent and its optimizer
    agt = ContinuousRPOAgt(env.obs_space, env.act_space // 2)  # Action space given to the environment is for two agents, but the Actor network should only return the action of one
    optimizer = torch.optim.Adam(agt.parameters(), lr=L_RATE, eps=1e-5)

    # Initialize the environment
    next_obs = env.reset()
    next_done = False  # Environment never starts in its final state

    for update in range(num_updates):
        # If required anneal the learning rate
        if ANNEAL_LR:
            optimizer.param_groups[0]['lr'] = (1 - (update / num_updates)) * L_RATE

        # Initialize the memory
        # TODO: Should memory persist over an agent's lifetime?
        memory = RLMemory()

        # Switch the agent to inference mode
        agt = agt.eval()

        # Gather experience over a num_steps
        for step in range(NUM_STEPS):
            # For each individual in the environment
            actions = []
            for obs in next_obs:
                # Store the current state and final flag
                memory.add_obs(obs, next_done)

                # Get the actions and values based on the current state
                with torch.no_grad():
                    action, logprob, _, value = agt.get_action_and_value(obs)

                # Store information related to action
                memory.add_act_and_val(action, logprob, value)
                actions.append(action.cpu().tolist())

            # Move a step froward
            # Note that the actions are reversed, since ind_1 is setting the acceleration for ind_2,
            # and vice versa.
            next_obs, reward, next_done = env.step(Action(actions[1], actions[0]))

            # Add the reward to memory twice (once for each individual)
            for _ in range(2):
                memory.add_rew(reward)


        # TODO: Training starts here
        pass

except KeyboardInterrupt:
    pass
finally:
    # Close the environment
    env.close()
