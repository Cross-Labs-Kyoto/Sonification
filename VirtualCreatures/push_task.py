#!/usr/bin/env python3
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from environments import PushEnv, Action
from agents import ContinuousRPOAgt
from utils import RLMemory

from loguru import logger



# Declare environment configuration
# TODO: This should become CLI parameters
HEADLESS = False
DT = 1 / 30
SIZE = 800
RND_SEED = 42
L_RATE = 3e-4
ANNEAL_LR = False
TOT_TIMESTEPS = 8000000
NUM_MINI_BATCHES = 32
NUM_STEPS = 2048 // 2  # Given that we have two agents in the environment, data will be gathered twice as fast
GAMMA = 0.99
GAE_LAMBDA = 0.95
NB_EPOCHS = 10
CLIP_COEF = 0.2
ENT_COEF = 0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
KL_TARG = 0.5

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
env = PushEnv(SIZE, DT, headless=HEADLESS)

try:
    # Instantiate the agent and its optimizer
    agt = ContinuousRPOAgt(env.obs_space, env.act_space // 2)  # Action space given to the environment is for two agents, but the Actor network should only return the action of one
    # TODO: If required load the agent
    optimizer = torch.optim.Adam(agt.parameters(), lr=L_RATE, eps=1e-5)

    # Initialize the environment
    next_obs = env.reset()
    next_done = 0  # Environment never starts in its final state

    ep_rew = 0
    nb_ep = 0
    for update in range(num_updates):
        logger.info(f'Update {update + 1}')
        # If required anneal the learning rate
        if ANNEAL_LR:
            optimizer.param_groups[0]['lr'] = (1 - (update / num_updates)) * L_RATE

        # Initialize memories for each agent
        # TODO: Should memory persist over an agent's lifetime?
        memories = [RLMemory() for _ in range(env.nb_agts)]

        # Switch the agent to inference mode
        agt = agt.eval()

        # Gather experience over a num_steps
        for step in range(NUM_STEPS):
            # For each individual in the environment
            actions = []
            for idx, obs in enumerate(next_obs):  # There are as many observations as there are individuals in the environment
                obs = torch.tensor(obs).flatten().unsqueeze(0)
                # Store the current state and final flag
                memories[idx].add_obs(obs, next_done)

                # Get the actions and values based on the current state
                with torch.no_grad():
                    action, logprob, _, value = agt.get_action_and_value(obs)

                # Store information related to action
                memories[idx].add_act_and_val(action, logprob, value)
                actions.append(action.squeeze().cpu().tolist())

            # Move a step froward
            # Note that the actions are reversed, since ind_1 is setting the acceleration for ind_2,
            # and vice versa.
            next_obs, reward, next_done = env.step(Action(*reversed(actions)))
            ep_rew += reward

            if next_done:
                # Reset the environment
                next_obs = env.reset()
                # Log the episodic reward when done
                nb_ep += 1
                logger.info(f'Episode {nb_ep} - Reward: {ep_rew}')
                ep_rew = 0

            # Assumes equal contribution from all agents
            # Therefore, they all get the same reward
            for mem in memories:
                mem.add_rew(reward)

        # Compute the advantageous returns based on the collected rewards and values
        for mem, obs in zip(memories, next_obs):
            mem.rewards = mem.rewards.to(agt.device)

            with torch.no_grad():
                obs = torch.tensor(obs).flatten().unsqueeze(0)
                next_value = agt.get_value(obs)
                last_gae_lam = 0
                mem.advantages = torch.zeros_like(mem.rewards).to(agt.device)

            # Compute the advantage for each state
            # TODO: Might have to adapt the indexing of advantages here
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - mem.dones[t+1].item()
                    nextvalues = mem.values[t+1]
                delta = mem.rewards[t] + GAMMA * nextvalues * nextnonterminal - mem.values[t]
                mem.advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * last_gae_lam

            # Compute the actual return
            mem.returns = mem.advantages + mem.values

        # Build DataLoader for training policy and value networks
        # Concatenate the data from all agent's memories
        dl = DataLoader(ConcatDataset(memories), batch_size=minibatch_size, num_workers=0)

        # Train policy and value networks
        agt = agt.train()
        for epoch in range(NB_EPOCHS):
            train_loss = 0
            for batch in dl:
                obs, acts, log_probs, rews, dones, values, advs, rets = batch
                _, new_log_probs, entropy, new_values = agt.get_action_and_value(obs, acts)  # TODO: Might have to squeeze results
                log_ratio = new_log_probs - log_probs
                ratio = log_ratio.exp()

                # The difference between the old and new policy distributions will determine when learning stops
                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                # Policy loss
                pg_loss_1 = -advs * ratio
                pg_loss_2 = -advs * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                # Value loss
                v_loss = 0.5 * ((new_values - rets) ** 2).mean()  # TODO: Check dimensions

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agt.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                train_loss += loss.item()

            logger.info(f'Epoch {epoch} - Loss: {train_loss / len(dl)} - KL-divergence: {approx_kl.item()}')

            # Early stopping
            if approx_kl < KL_TARG:
                break

except KeyboardInterrupt:
    pass
finally:
    # Close the environment
    env.close()
