#!/usr/bin/env python3
import hashlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class RLMemory(Dataset):
    """
    Aggregates experiences in the shape of state, action, reward and done/final.
    This is primarily intended for use in Reinforcement Learning processes.
    """

    def __init__(self):
        super().__init__()

        # Keep track of the state, actions, probabilities associated with each action, rewards, computed values, and flags indicating the end of a task
        self._obs = None
        self._acts = None
        self._log_probs = None
        self._rewards = None
        self._dones = None
        self._values = None


    def add(self, obs, acts, lprobs, rew, done, val):
        # Flatten everything and transform into tensors
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).flatten()
        else:
            obs = torch.tensor(obs).flatten()

        if isinstance(acts, np.ndarray):
            acts = torch.from_numpy(acts).flatten()
        else:
            acts = torch.tensor(acts).flatten()

        if isinstance(lprobs, np.ndarray):
            lprobs = torch.from_numpy(lprobs).flatten()
        else:
            lprobs = torch.tensor(lprobs).flatten()

        # This assumes that reward, done, and value are floats/bools instead of lists or numpy arrays
        rew = torch.tensor([rew])
        done = torch.tensor([done])
        val = torch.tensor([val])

        # Store everything in their respective sub-sets
        if self._obs is None:
            # Direct assignment
            self._obs = obs
            self._acts = acts
            self._log_probs = lprobs
            self._rewards = rew
            self._dones = done
            self._values = val
        else:
            # Stack at the bottom
            self._obs = torch.vstack([self._obs, obs])
            self._acts = torch.vstack([self._acts, acts])
            self._log_probs = torch.vstack([self._log_probs, lprobs])
            self._rewards = torch.vstack([self._rewards, rew])
            self._dones = torch.vstack([self._dones, done])
            self._values = torch.vstack([self._values, val])

    def __len__(self):
        # Assume that all subsets are of the same length
        return len(self._obs)

    def __getitem__(self, idx):
        return self._obs[idx], self._acts[idx], self._log_probs[idx], self._rewards[idx], self._dones[idx], self._values[idx]


class Memory(Dataset):
    """Aggregates unique experiences to use for training models."""

    def __init__(self):
        super().__init__()

        # Keep track of inputs added to the dataset to make sure they are unique
        self._in_set = set()
        self._data = None
        self._inpts = None
        self._targs = None

    def add(self, inpt, targ):
        """Adds an (input, target) pair to the dataset, if they are not already part of it."""

        # Computes a hash of the input
        inpt_hash = hashlib.sha256(inpt.tobytes(), usedforsecurity=False)

        # If not a duplicate
        if inpt_hash not in self._in_set:
            # Add the record to the dataset
            inpt = torch.tensor(inpt, dtype=torch.float32).unsqueeze(0)
            targ = torch.tensor(targ, dtype=torch.float32).unsqueeze(0)
            if self._inpts is None:
                self._inpts = inpt
                self._targs = targ
            else:
                self._inpts = torch.vstack([self._inpts, inpt])
                self._targs = torch.vstack([self._targs, targ])
            self._in_set.add(inpt_hash)

    def __len__(self):
        return len(self._in_set)

    def __getitem__(self, idx):
        # Return the corresponding input and target
        return self._inpts[idx], self._targs[idx]


class SoundMapper(nn.Module):
    """Defines a trainable mapping between xenobot movement features, and sound (more specifically, frequency and amplitude)."""

    def __init__(self, nb_ins: int, hidden_lays: list[int], nb_outs: int = 2, l_rate: float = 1e-3, device: str = 'cuda'):
        """Declares a multi-layer perceptron linking input features to frequency and amplitude.

        Parameters
        ----------
            nb_ins: int
                The number of inputs.

            hidden_lays: list[int]
                A list of sizes for the hidden layers. If empty, the input and output layers will be connected directly.

            nb_outs: int, optional
                The number of output layers.

            l_rate: float, optional
                The step size for updating the network's parameters.

            device: {'cuda', 'cpu', 'auto'}, optional
                The type of device to use for computation.

        """

        # Initialize the parent class
        super().__init__()

        # Define the device to use for computation
        if device in ['cuda', 'cpu']:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Define the network's architecture
        if len(hidden_lays) == 0:
            self._linear = nn.Sequential(
                nn.Linear(nb_ins, out_features=nb_outs, device=self._device),
                nn.Sigmoid()
            )
        else:
            self._linear = nn.Sequential()
            in_size = nb_ins
            for hid_size in hidden_lays:
                # Add layer
                self._linear.append(nn.Linear(in_size, hid_size, device=self._device))
                #self._linear.append(nn.ReLU(inplace=True))  # Could be replaced with Tanh()
                self._linear.append(nn.Tanh())
                self._linear.append(nn.LayerNorm(hid_size, device=self._device))
                # Record size for next input
                in_size = hid_size

            # Add output layer
            self._linear.append(nn.Linear(in_size, nb_outs, device=self._device))
            #self._linear.append(nn.Sigmoid())
            self._linear.append(nn.Hardsigmoid())

        # Define optimizer
        self.optim = torch.optim.SGD(self.parameters(), lr=l_rate, momentum=0.9, weight_decay=1e-5, maximize=False)  # Assumes we want to minimize the loss

    def forward(self, x):
        # Make sure the input is on the right device
        if x.device != self._device:
            x = x.to(self._device)

        # Propagate the input through the multi-layer perceptron
        out = self._linear(x).squeeze()

        # Scale and return the frequencies for X and Y coordinates
        return out

    def train_nn(self, weight_path, dset, loss_fn, nb_epoch, batch_size, patience: int = 5):
        """Uses the given dataset and loss function to train the model.
        """

        # Get a dataloader from the provided dataset
        dl = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)

        # Switch the model to training mode
        self = self.train()

        # Train the network
        min_loss = np.inf
        cnt = 0
        for epoch in range(nb_epoch):
            train_loss = 0
            for inpts, targs in dl:
                # Reset the gradient
                self.optim.zero_grad()

                # Extract the inputs and targets
                inpts = inpts.to(self._device)
                targs = targs.to(self._device)

                # Compute the predictions
                preds = self(inpts)

                # Compute and back-propagate loss
                loss = loss_fn(preds, targs)
                loss.backward()
                self.optim.step()

                # Aggregate the loss
                train_loss += loss.item()

            train_loss /= len(dl)

            # Save the weights if the loss decreased
            if train_loss < min_loss:
                min_loss = train_loss
                torch.save(self.state_dict(), weight_path)
                cnt = 0
            else:
                cnt += 1
                if cnt > patience:
                    # Stop training if the loss has been decreasing for some time
                    break

        # Return the loss
        logger.info(f'Min Loss {min_loss}')
        return min_loss

    @property
    def device(self):
        return self._device


class RecurrentSoundMapper(SoundMapper):
    """Defines a trainable recurrent mapping between xenobot movement features and sound."""

    def __init__(self, nb_ins: int, hidden_lays: list[int], nb_lstm: int, size_lstm: int,
                 nb_outs: int = 2, l_rate: float = 1e-3, device: str = 'cuda'):
        """Declares a lstm-based network linking input features to frequency and amplitude.

        Parameters
        ----------
            nb_ins: int
                The number of inputs.

            hidden_lays: list[int]
                A list of sizes for the hidden layers. If empty, the input and output layers will be connected directly.

            nb_lstm: int
                The number of lstm layers to include in the network.

            size_lstm: int
                The size of all lstm layers.

            nb_outs: int, optional
                The number of output layers.

            l_rate: float, optional
                The step size for updating the network's parameters.

            device: {'cuda', 'cpu', 'auto'}, optional
                The type of device to use for computation.

        """

        # Make sure the number and size of LSTM is non-zero
        assert nb_lstm != 0 and size_lstm != 0, 'The number and size of lstm layers cannot be zero. Use a standard `SoundMapper()` if this is what you want.'

        # Initialize the SoundMapper
        super().__init__(size_lstm, hidden_lays, nb_outs, l_rate, device)

        # Declare the recurrent portion of the network
        self._lstm = nn.LSTM(nb_ins, size_lstm, num_layers=nb_lstm, batch_first=True, device=self._device)

        # Redefine the optimizer
        self.optim = torch.optim.SGD(self.parameters(), lr=l_rate, momentum=0.9, weight_decay=1e-5, maximize=False)  # Assumes we want to minimize the loss


    def forward(self, x):
        # Make sure the input is on the same device as the model
        if x.device != self._device:
            x = x.to(self._device)

        # Propagate the input through the lstm
        # Discard the hidden and cell states
        out, _ = self._lstm(x)

        # Return the result of propagating the last hidden state through the linear portion
        return super().forward(out[:, -1, :])


class AttentionSoundMapper(SoundMapper):
    """Defines a trainable attention-based mapping between xenobot movement features and sound."""

    def __init__(self, nb_ins: int, hidden_lays: list[int], nb_heads: int, embed_size: int, nb_outs: int = 2,
                 l_rate: float = 1e-3, device: str = 'cuda'):
        """Declares an attention-based mapper linking input features to frequency and amplitude.

        Parameters
        ----------
            nb_ins: int
                The number of inputs.

            hidden_lays: list[int]
                A list of sizes for the hidden layers. If empty, the input and output layers will be connected directly.

            nb_heads: int
                The number of attention heads to use. Keep in mind that each attention head will have a dimension of `embed_size // nb_heads`.

            embed_size: int
                The size of the embedding vector.
            nb_outs: int, optional
                The number of output layers.

            l_rate: float, optional
                The step size for updating the network's parameters.

            device: {'cuda', 'cpu', 'auto'}, optional
                The type of device to use for computation.

        """

        # Initialize the parent SoundMapper
        super().__init__(embed_size, hidden_lays, nb_outs, l_rate, device)

        # Define embedding and attention layer
        # Assumes an MLP embedding layer
        self._embed = nn.Sequential(
            nn.Linear(nb_ins, embed_size, device=self._device),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_size, device=self._device)
        )

        self._attn = nn.MultiheadAttention(embed_size, nb_heads, batch_first=True, device=self._device)

        # Update the optimizer's definition
        self.optim = torch.optim.SGD(self.parameters(), lr=l_rate, momentum=0.9, weight_decay=1e-5, maximize=False)  # Assumes we want to minimize the loss

    def forward(self, x):
        # Make sure the input is on the same device as the model
        if x.device != self._device:
            x = x.to(self._device)

        # Propagate the input through the embedding and attention layers
        out = self._attn(self.embed(x))

        # Return the frequency and amplitude
        return super().forward(out)
