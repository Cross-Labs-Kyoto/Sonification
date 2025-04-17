#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Process, Event, Queue
from queue import Empty

import numpy as np
import torch
from torch import nn
import cv2 as cv
import cellpylib as cpl
from pedalboard.io import AudioFile, AudioStream
import librosa
from loguru import logger

from utils import SoundMapper
from settings import ROOT_DIR


END_EVT = Event()
DISP_Q = Queue()
AUDIO_Q = Queue()
LOOP_LEN = 1 / 30  # Corresponds to a 30 fps frame rate
BATCH_SIZE = 32
PATIENCE = 5

class MarcoPoloRule(cpl.BaseRule):
    """Defines a custom rule for setting the state of CA cells based on the frequencies received."""

    def __init__(self, init_poses):
        self.agt_poses = np.array(init_poses).astype(int)

    def __call__(self, n, c, t):
        # Activate the cells corresponding to the next_poses only
        # X and Y are reversed because I am a moron and didn't account for that everywhere else
        # At least the display is consistent with what one would expect
        if np.any(np.all(c[::-1] == self.agt_poses, axis=1)):
            return 0
        else:
            # Leave the rest inactive
            return 1


def display():
    win_name = 'Debug'

    # Declare a named window
    cv.namedWindow(win_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)

    try:
        while not END_EVT.wait(LOOP_LEN):
            # Get next state from queue
            try:
                state = DISP_Q.get(block=False)
            except Empty:
                continue
            except ValueError:
                break

            # Display state
            # Scale the environment up so that we can see things happening
            frame = cv.resize(state.astype(np.float32), None, fx=10, fy=10, interpolation=cv.INTER_NEAREST)
            cv.imshow(win_name, frame)
            cv.pollKey()

    except KeyboardInterrupt:
        pass
    finally:
        # Close display window
        cv.destroyWindow(win_name)

        if not END_EVT.is_set():
            END_EVT.set()


def audio(out_path, sample_rate):
    """Plays stereo audio chunks."""

    # Initialize the audio output 
    audio_file = Path(out_path).expanduser().resolve()
    if audio_file.is_file():
        audio_kwargs = dict(filename=audio_file, mode='w', samplerate=sample_rate)
        audio_cls = AudioFile
    else:
        audio_kwargs = dict(output_device_name=out_path, num_output_channels=2)
        audio_cls= AudioStream

    # Pre-compute logarithmic amplitude envelop to be applied to all audio chunks to avoid clipping
    fade_len = int(sample_rate * LOOP_LEN / 10)
    full_len = int(sample_rate * LOOP_LEN) - 2 * fade_len
    amp = np.log(np.linspace(1, np.exp(1), fade_len))
    amp = np.hstack([amp, np.ones((full_len, )), amp[::-1]]).astype(np.float32)

    # Generate stereo tones according to received frequencies
    try:
        with audio_cls(**audio_kwargs) as audio_out:
            while not END_EVT.wait(LOOP_LEN):
                # Get frequencies
                try:
                    freqs = AUDIO_Q.get(block=False)
                except Empty:
                    continue

                # Generate the corresponding tones
                tones = []
                for freq in freqs:
                    tone = librosa.tone(freq, sr=sample_rate, length=int(sample_rate * LOOP_LEN))
                    tones.append(tone * amp)

                # Mix tones
                tones = np.array(tones).reshape((2, 2, -1))
                chunk = tones.sum(axis=1).astype(np.float32)

                # Write chunks to audio stream
                if isinstance(audio_out, AudioStream):
                    audio_out.write(chunk, sample_rate=sample_rate)
                else:
                    audio_out.write(chunk)
    except KeyboardInterrupt:
        pass
    finally:
        if not END_EVT.is_set():
            END_EVT.set()


if __name__ == '__main__':
    # Declare a command line interface
    parser = ArgumentParser()
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='A flag indicating whether to run the simulation in debug mode or not.')
    parser.add_argument('-s', '--size', dest='ca_size', type=int, default=50,
                        help='The size of one side of the environment in cells.')
    parser.add_argument('-a', '--audio', dest='audio_dev', type=str,
                        default='Default ALSA Output (currently PipeWire Media Server)',
                        help='Defines a path where audio should be written, or the audio device to use as output. This parameter is only valid when used with --debug. Otherwise, it is ignored.')
    parser.add_argument('-r', '--sample_rate', dest='sr', type=int, default=44100,
                        help='The sample rate for the audio output. This parameter is only valid when used with --debug. Otherwise, it is ignored.')
    parser.add_argument('-m', '--freq_min', dest='fmin', type=int, default=200,
                        help='The minimum frequency to use when generating tones.')
    parser.add_argument('-M', '--freq_max', dest='fmax', type=int, default=3200,
                        help='The maximum frequency to use when generating tones.')
    # TODO: Provide parameter to load weights from file
    # TODO: Provide parameter to specify learning rate

    # Parse the command line arguments
    args = parser.parse_args()

    assert args.fmin < args.fmax, f'The minimum frequency has to be less than the maximum, but got: {args.fmin} and {args.fmax}'

    # Initialize mapper, associated loss, and optimizer
    mapper = SoundMapper(nb_ins=2, hidden_lays=[50, 50], l_rate=1e-4)
    loss_fn = nn.MSELoss()

    # Initialize required constants
    bins = np.linspace(args.fmin, args.fmax, args.ca_size, dtype=np.float32)
    bins = torch.from_numpy(np.stack([bins, bins], axis=0)).to(mapper.device)

    # Start the display and audio threads if necessary
    if args.debug:
        threads = [Process(target=display),
                   Process(target=audio, args=(args.audio_dev, args.sr))]
        for thr in threads:
            thr.start()

        queues = [DISP_Q,
                  AUDIO_Q]

    try:
        # Initialize the list of predicted and target frequencies
        preds = []
        targs = []

        # Keep track of the minimum loss
        min_loss = np.inf


        # TODO: Loop over envs until KeyboardInterrupt
        # TODO: Stop when loss stops decreasing (after some patience)

        # Initialize the marco-polo rule
        init_poses = np.random.randint(low=0, high=args.ca_size-1, size=(2, 2))
        mp_rule = MarcoPoloRule(init_poses)

        if args.debug:
            # Initialize the CA environment with two agents
            ca_hist = np.ones((1, args.ca_size, args.ca_size), dtype=int)
            ca_hist[:, init_poses[0, 0], init_poses[0, 1]] = 1
            ca_hist[:, init_poses[1, 0], init_poses[1, 1]] = 1

        while np.any(mp_rule.agt_poses[0] != mp_rule.agt_poses[1]) and not END_EVT.is_set():
            freqs = []
            for pos in mp_rule.agt_poses:
                # Generate the frequencies corresponding to both agents
                preds.append(mapper(torch.tensor(pos, dtype=torch.float32)))
                freqs.append(preds[-1] * (args.fmax - args.fmin) + args.fmin)

                # Get target frequencies
                targs.append((bins[(0, 1), pos] - args.fmin) / (args.fmax - args.fmin))
                # logger.debug(f'{pos} => {preds[-1]} <> {targs[-1]}')

            # Update position of both agents based on received frequencies
            com_pos_1 = torch.argmin(torch.abs(bins - freqs[-2].unsqueeze(1)), dim=1)
            com_pos_2 = torch.argmin(torch.abs(bins - freqs[-1].unsqueeze(1)), dim=1)

            # TODO: Make sure this is correct (see commented logger above, and comment in display function)
            diff = com_pos_1 - com_pos_2
            abs_diff= torch.abs(diff).tolist()
            sign_diff = torch.sign(diff).tolist()
            if abs_diff[0] >= abs_diff[1]:
                mp_rule.agt_poses[0][0] -= 1
                mp_rule.agt_poses[1][0] += 1
            else:
                mp_rule.agt_poses[0][1] -= 1
                mp_rule.agt_poses[1][1] += 1

            # Wrap around if agents go over the edge
            mp_rule.agt_poses = np.where(mp_rule.agt_poses < 0, args.ca_size - 1, mp_rule.agt_poses)
            mp_rule.agt_poses = np.where(mp_rule.agt_poses >= args.ca_size, 0, mp_rule.agt_poses)

            if args.debug:
                # Update CA environment
                # Timesteps is set to 2 since the initial state is left unchanged in step 1
                ca_hist = cpl.evolve2d(ca_hist, timesteps=2, apply_rule=mp_rule)
                # Send data to display and audio queues
                DISP_Q.put(ca_hist[-1])
                aud_freqs = np.array([f.detach().cpu().numpy() for f in freqs]).reshape((4, 1)) 
                AUDIO_Q.put(aud_freqs)
                # Only keep the last CA state
                ca_hist = np.expand_dims(ca_hist[-1], axis=0)

            if len(targs) >= BATCH_SIZE:
                # Train the mapper
                batch_preds = torch.stack(preds, dim=0)
                batch_targs = torch.stack(targs, dim=0)

                loss = loss_fn(batch_preds, batch_targs)
                loss.backward()
                mapper.optim.step()

                loss = loss.item()
                logger.info(f'Loss: {loss}')

                # Reset the gradients
                mapper.optim.zero_grad()

                # Reset predictions and targets
                preds = []
                targs = []

                if loss < min_loss:
                    min_loss = loss
                    # Save weights
                    torch.save(mapper.state_dict(), ROOT_DIR.joinpath('ca_2_sound.pt'))

    except KeyboardInterrupt:
        pass

    finally:
        # Stop the display and audio threads if in debug mode
        if args.debug:
            if not END_EVT.is_set():
                END_EVT.set()

            for q in queues:
                while not q.empty():
                    try:
                        q.get(block=False)
                    except (Empty, ValueError):
                        break
                if hasattr(q, 'close'):
                    q.close()

            for thr in threads:
                thr.join()
