#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from collections import deque
import numpy as np
import torch
import librosa
from pedalboard.io import AudioFile, AudioStream
from loguru import logger

from utils import MvTracker, get_video_meta, VideoIterator, RecurrentSoundMapper
from settings import ROOT_DIR


# Define a command line interface
parser = ArgumentParser()
parser.add_argument('filename', type=str,
                    help='Either the name of a video file to process, or a camera ID.')
parser.add_argument('-t', '--thres', dest='canny_thres', type=int, default=40)
parser.add_argument('-o', '--output', dest='out', type=str, default='default',
                    help='Either the relative path to a file or the name of a device to which the audio will be written.')
parser.add_argument('-w', '--weight', dest='load', type=str,
                    help="The relative path to the file from which to load the model's parameters.")
parser.add_argument('-l', '--learn_rate', dest='l_rate', type=float, default=1e-3,
                    help="The length of the step for each iteration of the gradient descent algorithm.")
parser.add_argument('-v', '--volume', dest='vol', type=float, default=0.5,
                    help="The volume in percentage at which to output audio. Defaults to: 0.5")
parser.add_argument('-r', '--sample_rate', dest='sample_rate', type=int, default=44100,
                    help="The sample rate to use for the audio output.")
parser.add_argument('-s', '--skip', dest='f_skip', type=float, default=0,
                    help='The number of seconds to skip between tracking events. The provided value should be a float > 0.')
parser.add_argument('--list', dest='lst_devices', action='store_true',
                    help='List all available output audio devices.')

# And parse all command line arguments
args = parser.parse_args()

# If requested, list the output audio devices
if args.lst_devices:
    logger.info('The following output devices are available:')
    for dev_name in AudioStream.output_device_names:
        logger.info(dev_name)

    # And exit
    exit(0)

# Define where the audio output should go
if args.out in AudioStream.output_device_names:
    audio_out_cls = AudioStream
    audio_out_kwargs = dict(output_device_name=args.out, num_output_channels=1, sample_rate=args.sample_rate) 
else:
    fname = Path(args.out).expanduser().resolve()
    audio_out_cls = AudioFile
    audio_out_kwargs = dict(filename=str(fname), samplerate=args.sample_rate, num_channels=1)

# Define where to take the video input from
try:
    video_in = int(args.filename)
except ValueError:
    video_in = Path(args.filename).expanduser().resolve()
    if not video_in.is_file():
        raise RuntimeError(f'The provided video file does not exist or is not a file: {video_in}')
    video_in = str(video_in)

# Instantiate a sound mapper
model = RecurrentSoundMapper(nb_ins=1, hidden_lays=[5, 5], nb_lstm=1, size_lstm=10, l_rate=args.l_rate, fmin=500, fmax=2000, amin=0, amax=args.vol, device='auto')

# Load the parameters if necessary
if args.load is not None:
    weight_file = Path(args.load).expanduser().resolve()
    if weight_file.is_file():
        state_dict = torch.load(weight_file, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        logger.error(f'The provided weight file, does not exist or is not a file: {weight_file}')

# TODO: Declare the loss
Loss = None

# Loop forever (or until the user presses ctrl+c and asks to quit the program)
while True:
    with audio_out_cls(**audio_out_kwargs) as audio_out:
        with VideoIterator(video_in) as vi:
            # Get the video metadata
            vid_w, vid_h, fps, tot_frames = get_video_meta(vi)

            # Instantiate a new movement tracker
            tracker = MvTracker(vid_w, vid_h, max_dist=560, offset_x=50, canny_thres=args.canny_thres)
            
            # Rest the gradient
            model.optim.zero_grad()

            # Pre-compute the amplitude to apply to all audio chunks
            # This is to prevent any clipping noise between chunks
            fade_len = int(args.sample_rate/(10*fps))
            full_len = int(args.sample_rate/fps) - 2 * fade_len
            log_amp = np.log(np.linspace(1, np.exp(1), fade_len))
            log_amp = np.hstack([log_amp, np.ones((full_len, )), log_amp[::-1]]).astype(np.float32)

            # Process the video and generate audio output
            speeds = deque(maxlen=30)
            preds = []
            try:
                for f_idx, frame in enumerate(vi):
                    # If requested skip frames
                    if args.f_skip != 0:
                        if f_idx % int(args.f_skip * fps) != 0:
                            continue

                    # Track objects
                    tracker.track(frame)

                    # We assume that only a single object is tracked
                    # TODO: If assumption is not correct, then manage a list of models, and save their parameters accordingly
                    try:
                        obj = tracker.tracked_objects[0]
                    except IndexError:
                        continue

                    if obj.id in tracker.rel_pos:
                        # Keep track of the agent's rotational speed over a length of time
                        old_pos, curr_pos = tracker.rel_pos[obj.id]
                        speeds.append((curr_pos[1] - old_pos[1]) * fps)
                        
                        # Format the input and compute the frequency and amplitude
                        in_data = np.array(speeds).reshape((1, -1, 1)).astype(np.float32)
                        freq, amp = model(torch.from_numpy(in_data))

                        # Generate the associated audio chunk
                        chunk = librosa.tone(frequency=freq.detach().cpu().item(), sr=args.sample_rate, length=int(args.sample_rate/fps)).astype(np.float32)

                        # Store the list of outputs
                        preds.append((freq, amp))

                        # Play the sound
                        if isinstance(audio_out, AudioStream):
                            audio_out.write(chunk * amp.detach().cpu().item() * log_amp, sample_rate=args.sample_rate)
                        else:
                            audio_out.write(chunk * amp.detach().cpu().item() * log_amp)
                            
            except KeyboardInterrupt:
                pass
            finally:
                # Compute loss and optimize parameters
                # TODO: provide the necessary targets to the loss function
                print(preds)
                loss = Loss(preds, targ)
                loss.backward()
                model.optim.step()

                # Save parameters
                state_dict = model.state_dict()
                torch.save(state_dict, ROOT_DIR.joinpath('mv_2_sound.pt'))

                # Pause until an answer is given
                res = input('Do you want to quit the program (y/N): ')
                if res.lower() == 'y':
                    break

