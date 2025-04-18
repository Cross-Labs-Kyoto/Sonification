#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import Process, Event, Queue
from queue import Empty

import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch import nn
import cv2 as cv
import cellpylib as cpl
from pedalboard.io import AudioFile, AudioStream
import librosa
from loguru import logger

from settings import ROOT_DIR
from utils import SoundMapper, Memory


END_EVT = Event()
DISP_Q = Queue()
AUDIO_Q = Queue()
LOOP_LEN = 1 / 30  # Corresponds to a 30 fps frame rate
NB_EPOCH = 200
BATCH_SIZE = 32

class MarcoPoloRule(cpl.BaseRule):
    """Defines a custom rule for setting the state of CA cells based on the frequencies received."""

    def __init__(self, init_poses):
        self.agt_poses = np.array(init_poses).astype(int)

    def __call__(self, n, c, t):
        # Activate the cells corresponding to the next_poses only
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
    parser.add_argument('-w', '--weights', dest='weight_file', type=str, default=None,
                        help='The relative path to the weight file to use for the mapper.')
    parser.add_argument('-l', '--learn_rate', dest='lr', type=float, default=1e-3,
                        help="The step size to use when updating the model's parameters.")
    parser.add_argument('-p', '--exploration_proba', dest='init_exp_proba', type=float, default=1,
                        help="Sets the initial exploration probability.")
    parser.add_argument('-t', '--test', dest='testing', action='store_true',
                        help='A flag indicating whether to run the simulation in testing mode, in which no learning occurs, or not.')
    parser.add_argument('-c', '--criterion', dest='loss_crit', type=float, default=0.0005,
                        help="Defines a threshold loss under which the algorithm stops learning.")

    # Parse the command line arguments
    args = parser.parse_args()

    assert args.fmin < args.fmax, f'The minimum frequency has to be less than the maximum, but got: {args.fmin} and {args.fmax}'

    # Initialize mapper, associated loss, and optimizer
    mapper = SoundMapper(nb_ins=2, hidden_lays=[5, 5, 5], l_rate=args.lr)
    loss_fn = nn.MSELoss()

    # If provided load weights
    if args.weight_file is not None:
        weight_file = Path(args.weight_file).expanduser().resolve()
        if weight_file.is_file():
            mapper.load_state_dict(torch.load(weight_file, map_location=mapper.device, weights_only=True))
        else:
            logger.error(f'The provided path for the weights does not exist or is not a file: {weight_file}')
    else:
        weight_file = ROOT_DIR.joinpath('Models', 'ca_2_sound.pt')

    # Ensure the mapper is in inference mode
    mapper = mapper.eval()

    # Display mapper's topology
    logger.info(mapper)

    # Initialize a dataset to aggregate experience and train the mapper
    memory = Memory()

    # Initialize required constants
    bins = np.linspace(args.fmin, args.fmax, args.ca_size, dtype=np.float32)
    bins = torch.from_numpy(np.stack([bins, bins], axis=0))

    # Start the display and audio threads if necessary
    if args.debug:
        threads = [Process(target=display),
                   Process(target=audio, args=(args.audio_dev, args.sr))]
        for thr in threads:
            thr.start()

        queues = [DISP_Q,
                  AUDIO_Q]

    try:
        testing = args.testing
        exploration_proba = max(0, min(1, args.init_exp_proba))
        decay_rate = 0.005
        cool_down = 50
        while not END_EVT.is_set():
            try:
                # Initialize the marco-polo rule
                init_poses = np.random.randint(low=0, high=args.ca_size-1, size=(2, 2))
                mp_rule = MarcoPoloRule(init_poses)

                if args.debug:
                    # Initialize the CA environment with two agents
                    ca_hist = np.ones((1, args.ca_size, args.ca_size), dtype=int)
                    ca_hist[:, init_poses[0, 0], init_poses[0, 1]] = 1
                    ca_hist[:, init_poses[1, 0], init_poses[1, 1]] = 1

                while cdist(np.expand_dims(mp_rule.agt_poses[0], axis=0), np.expand_dims(mp_rule.agt_poses[1], axis=0)) > np.sqrt(2) and not END_EVT.is_set():
                    freqs = []
                    with torch.no_grad():
                        for pos in mp_rule.agt_poses:
                            # Generate the frequencies corresponding to both agents
                            pred = mapper(torch.tensor(pos, dtype=torch.float32)).cpu()
                            freqs.append(pred * (args.fmax - args.fmin) + args.fmin)

                            # Get target frequencies
                            targ = (bins[(0, 1), pos] - args.fmin) / (args.fmax - args.fmin)

                            # Add the experience to memory
                            memory.add(pos, targ)

                    # Update position of both agents based on received frequencies
                    com_pos_1 = torch.argmin(torch.abs(bins - freqs[0].unsqueeze(1)), dim=1)
                    com_pos_2 = torch.argmin(torch.abs(bins - freqs[1].unsqueeze(1)), dim=1)

                    # TODO: Uncomment if: Allow movement in only 4 directions
                    #diff = com_pos_1 - com_pos_2
                    #abs_diff = torch.abs(diff).tolist()
                    #sign_diff = torch.sign(diff).tolist()
                    #
                    # Prevent agents from staying in place
                    #if np.all(sign_diff == 0):
                    #    sign_diff = np.ones_like(sign_diff)
                    #if abs_diff[0] >= abs_diff[1]:
                    #    mp_rule.agt_poses[0][0] -= sign_diff[0]
                    #    mp_rule.agt_poses[1][0] += sign_diff[0]
                    #else:
                    #    mp_rule.agt_poses[0][1] -= sign_diff[1]
                    #    mp_rule.agt_poses[1][1] += sign_diff[1]

                    # Allow movement in 8 directions
                    sign_diff = torch.sign(com_pos_1 - com_pos_2).numpy()

                    # At random points in time
                    if not testing and np.random.random() < exploration_proba:
                        # Move agents in a random direction
                        sign_diff = np.random.choice([-1, 0, 1], size=sign_diff.shape[0])
                        # Update the exploration probability
                        exploration_proba *= (1 - decay_rate)

                    mp_rule.agt_poses[0] -= sign_diff
                    mp_rule.agt_poses[1] += sign_diff

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

                    cool_down -= 1
                    # Once enough memories have been added
                    if not testing and len(memory) % BATCH_SIZE == 0 and cool_down <= 0:
                        cool_down = 50
                        # Train the mapper
                        loss = mapper.train_nn(weight_file, memory, loss_fn, NB_EPOCH, BATCH_SIZE, patience=int(NB_EPOCH * 0.05))

                        # If model is sufficiently trained
                        if loss <= args.loss_crit:
                            # Switch to testing mode
                            testing = True
                            if not args.debug:
                                logger.info('Training complete.')
                                exit()
                            else:
                                logger.info('Training complete. Switching to testing mode.')

                        # Switch the mapper back to inference mode
                        mapper = mapper.eval()

            except KeyboardInterrupt:
                break
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
