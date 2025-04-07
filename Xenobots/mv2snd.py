#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from pedalboard.io import AudioFile, AudioStream
from loguru import logger

from utils import MvTracker, get_video_meta, VideoIterator
from settings import ROOT_DIR


class Speed2SoundModel(object):
    """Define a mapping taking rotation speed as input and translate it to frequency."""

    def __init__(self, f_min=20, f_max=20000, decay=0.1):
        super().__init__()

        self._f_min = torch.tensor(f_min, requires_grad=True)
        self._f_max = torch.tensor(f_max, requires_grad=True)
        self._decay = torch.tensor(decay, requires_grad=True)

        self._old_freq = None

    def __call__(self, theta):
        """It is assumed that the input is a rotational speed in radians per second."""

        # Make sure the input is a tensor
        theta = torch.tensor(theta)
        
        # Compute the current movement frequency
        mv_freq = theta / (2 * np.pi)

        # Apply decay if possible
        if self._old_freq is not None:
            mv_freq = self._old_freq * (1 - self._decay) + mv_freq * self._decay

        # Store the movement frequency for later computation
        self._old_freq = mv_freq

        # Translate the movement frequency into a sound frequency in the interval [f_min, f_max]
        return F.sigmoid(mv_freq) * (self._f_max - self._f_min) + self._f_min

    def reset(self):
        self._old_freq = None

    def parameters(self):
        return [self._f_min, self._f_max, self._decay]

    def state_dict(self):
        return dict(f_min=self._f_min, f_max=self._f_max, decay=self._decay)

    def load_state_dict(self, params):
        for k, v in params:
            # Translate the attribute name to make it "private"
            k = f'_{k}'
            # Only set valid attributes, the rest is gracefully ignored
            if hasattr(self, k):
                setattr(self, k, v)


if __name__ == "__main__":
    # Define a command line interface
    parser = ArgumentParser()
    parser.add_argument('filename', type=str, help='Either the name of a video file to process, or a camera ID.')
    parser.add_argument('-t', '--thres', dest='canny_thres', type=int, default=40)
    parser.add_argument('-o', '--output', dest='out', type=str, default='default',
                        help='Either the relative path to a file or the name of a device to which the audio will be written.')
    parser.add_argument('-w', '--weight', dest='load', type=str,
                        help="The relative path to the file from which to load the model's parameters.")
    parser.add_argument('-l', '--learn_rate', dest='l_rate', type=float, default=1e-3,
                        help="The length of the step for each iteration of the gradient descent algorithm.")
    parser.add_argument('-v', '--volume', dest='vol', type=float, default=0.5,
                        help="The volume in percentage at which to output audio. Defaults to: 50%")
    parser.add_argument('-r', '--sample_rate', dest='sample_rate', type=int, default=44100,
                        help="The sample rate to use for the audio output.")
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
        video_in = str(Path(args.filename).expanduser().resolve())

    # Instantiate model
    model = Speed2SoundModel(f_min=np.random.uniform(low=20, high=500), f_max=np.random.uniform(low=500, high=20000), decay=np.random.random())
    
    # Load the parameters if necessary
    if args.load is not None:
        weight_file = Path(args.load).expanduser().resolve()
        if weight_file.is_file():
            state_dict = torch.load(weight_file, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            logger.error(f'The provided weight file, does not exist or is not a file: {weight_file}')

    # Declare the optimizer
    # TODO: Modify the parameters to suit your needs
    optimizer = torch.optim.SGD(model.parameters(), args.l_rate, maximize=False)

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
                
                # Rest the gradient and the model
                model.reset()
                optimizer.zero_grad()

                # Pre-compute the amplitude to apply to all audio chunks
                # This is to prevent any clipping noise between chunks
                fade_len = int(args.sample_rate/(10*fps))
                full_len = int(args.sample_rate/fps) - 2 * fade_len
                amp = np.log(np.linspace(1, np.exp(1), fade_len))
                amp = np.hstack([amp, np.ones((full_len, )), amp[::-1]]).astype(np.float32) * args.vol

                # Process the video and generate audio output
                freqs = []
                try:
                    for frame in vi:
                        # Track objects
                        tracker.track(frame)

                        # We assume that only a single object is tracked
                        # TODO: If assumption is not correct, then manage a list of models, and save their parameters accordingly
                        try:
                            obj = tracker.tracked_objects[0]
                        except IndexError:
                            continue

                        if obj.id in tracker.rel_pos:
                            # Compute the frequency
                            old_pos, curr_pos = tracker.rel_pos[obj.id]
                            theta = (curr_pos[1] - old_pos[1]) * fps
                            freq = model(theta)

                            # Generate the associated audio chunk
                            chunk = librosa.tone(frequency=freq.detach().cpu().item(), sr=args.sample_rate, length=int(args.sample_rate/fps)).astype(np.float32)

                            # Store the list of local frequencies
                            freqs.append(freq)

                            # Play the sound
                            audio_out.write(chunk * amp, sample_rate=args.sample_rate)
                                
                except KeyboardInterrupt:
                    pass
                finally:
                    # Compute loss and optimize parameters
                    # TODO: provide the necessary targets to the loss function
                    loss = Loss(freqs, targ)
                    loss.backward()
                    optimizer.step()

                    # Save parameters
                    state_dict = model.state_dict()
                    torch.save(state_dict, ROOT_DIR.joinpath('speed_2_sound.pt'))

                    # Pause until an answer is given
                    res = input('Do you want to quit the program (y/N): ')
                    if res.lower() == 'y':
                        break

