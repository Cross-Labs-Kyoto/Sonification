#!/usr/bin/env python3
from pathlib import Path
import torch
from torch import nn
from torch.distributions.normal import Normal


class ContinuousRPOAgt(nn.Module):
    """
    Implements a Robust Policy Optimization (RPO) agent, for continuous (action, state) spaces.
    Heavily inspired from: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/rpo_continuous_action.py
    """

    def __init__(self, in_size, out_size, rpo_alpha=0.5, device='auto'):
        """
        Initializes the critic and both parts of the actor networks.

        Parameters
        ----------
        in_size: int
            The size of the state input fed to both the actor and critic.

        out_size: int
            The size of the Actor's output layer (= action).

        rpo_alpha: float, optional
            A ratio in the range [0, 1], which determines the amount of random perturbation on the action mean.
            Default to 0.5, which is equivalent to or better than PPO. After fine-tuning, can be decreased to 0.01.

        device: str
            The device to use for computations.

        """

        super().__init__()

        # Set required parameters
        self.rpo_alpha = rpo_alpha
        if device == 'auto':
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self._device = torch.device(device)

        # Define Actor and Critic topologies
        self._critic = nn.Sequential(
            nn.Linear(in_size, 64, device=self._device),
            nn.Tanh(),
            nn.Linear(64, 64, device=self._device),
            nn.Tanh(),
            nn.Linear(64, 1, device=self._device),
        )

        self._actor_mean = nn.Sequential(
            nn.Linear(in_size, 64, device=self._device),
            nn.Tanh(),
            nn.Linear(64, 64, device=self._device),
            nn.Tanh(),
            nn.Linear(64, out_size, device=self._device),
            nn.Tanh(),
        )

        # TODO: Why implement the action std as simple learnable parameters, instead of having a two headed Actor?
        self._actor_logstd = nn.Parameter(torch.zeros(1, out_size)).to(self._device)

    @property
    def device(self):
        return self._device

    def get_value(self, x):
        x = x.to(self._device)
        # And return the estimated value
        return self._critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.to(self._device)
        action_mean = self._actor_mean(x)
        action_logstd = self._actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            # This means that actions are not in the range [-1, 1]
            action = probs.sample()
        else:
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(self._device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self._critic(x)


    def save(self, fpath):
        """Saves the model's parameters to the given `fpath`.

        Parameters
        ----------
        fpath: pathlib.Path, str
            The relative path to the file in which to save the parameters.

        """

        if isinstance(fpath, str):
            fpath = Path(fpath).expanduser().resolve()

        torch.save(self.state_dict(), fpath)

    def load(self, fpath):
        """Loads the model's parameters from the given `fpath`.

        Parameters
        ----------
        fpath: pathlib.Path, str
            The relative path to the file from which to load the parameters.

        """

        if isinstance(fpath, str):
            fpath = Path(fpath).expanduser().resolve()

        if not fpath.is_file():
            raise RuntimeError(f'The provided path either does not exist or is not a file: {fpath}')

        self.load_state_dict(torch.load('model_weights.pth', weights_only=True))
        self.eval()
