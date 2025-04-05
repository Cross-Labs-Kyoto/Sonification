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
    parser.add_argument('-o', '--output', dest='out', type=str, required=True, help='Either the relative path to a file or the name of a device to which the audio will be written.')
    parser.add_argument('-w', '--weight', dest='load', type=str, help="The relative path to the file from which to load the model's parameters.")
    parser.add_argument('-l', '--learn_rate', dest='l_rate', type=float, help="The length of the step for each iteration of the gradient descent algorithm.", default=1e-3)
    parser.add_argument('-r', '--sample_rate', dest='sample_rate', type=int, help="The sample rate to use for the audio output.", default=44100)

    # And parse all command line arguments
    args = parser.parse_args()

    # Define where the audio output should go
    if args.out in AudioStream.output_device_names:
        audio_out = AudioStream(output_device_name=args.out, num_output_channels=1, sample_rate=args.sample_rate)
    else:
        fname = Path(args.out).expanduser().resolve()
        audio_out = AudioFile(str(fname), samplerate=args.sample_rate, num_channels=1)

    # Define where to take the video input from
    try:
        video_in = int(args.filename)
    except ValueError:
        video_in = str(Path(args.filename).expanduser().resolve())

    # Instantiate model
    model = Speed2SoundModel(f_min=100, f_max=300)
    
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
        with VideoIterator(video_in) as vi:
            # Get the video metadata
            vid_w, vid_h, fps, tot_frames = get_video_meta(vi)

            # Instantiate a new movement tracker
            tracker = MvTracker(vid_w, vid_h, max_dist=560, offset_x=50, canny_thres=args.canny_thres)
            
            # Rest the gradient and the model
            model.reset()
            optimizer.zero_grad()

            # Process the video and generate audio output
            global_freqs = []
            try:
                for frame in vi:
                    # Track objects
                    tracker.track(frame)

                    # TODO: Can we assume that only a single object will be present?
                    local_freqs = []
                    chunks = []
                    for obj in tracker.tracked_objects:
                        if obj.id in tracker.rel_pos:
                            # Compute the frequency
                            old_pos, curr_pos = tracker.rel_pos[obj.id]
                            theta = (old_pos[1] - curr_pos[1]) * fps
                            freq = model(theta)
                            local_freqs.append(freq)

                            # Generate the associated audio chunk
                            chunks.append(librosa.tone(freq, sr=args.sample_rate, length=int(args.sample_rate/fps)))

                    # Store the list of local frequencies
                    global_freqs.append(local_freqs)

                    # Mix the chunks together
                    chunks = np.array(chunks).mean(axis=0).astype(np.float32)

                    # Apply fade in/out to avoid clipping noises
                    # TODO: If too aggressive, find a better way to fade
                    amp = np.log(np.linspace(1, np.exp(1), num=int(args.sample_rate / (2 * fps))))
                    chunks *= np.hstack([amp, amp[::-1]])

                    # Play the sound
                    audio_out.write(chunks)
                            
            except KeyboardInterrupt:
                pass
            finally:
                # Compute loss and optimize parameters
                # TODO: provide the necessary targets to the loss function
                loss = Loss(global_freqs, targ)
                loss.backward()
                optimizer.step()

                # Save parameters
                state_dict = model.state_dict()
                torch.save(state_dict, ROOT_DIR.joinpath('speed_2_sound.pt'))

                # Pause until an answer is given
                res = input('Do you want to quit the program (y/N): ')
                if res.lower() == 'y':
                    break

