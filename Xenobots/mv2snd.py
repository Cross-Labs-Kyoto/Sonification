#!/usr/bin/env python3
import numpy as np
from scipy.io import wavfile
import cv2 as cv
import librosa
from tqdm import tqdm

from utils import MvTracker, cart_to_polar, get_video_meta
from settings import SAMPLE_RATE, DATA_DIR


# Instantiate a video reader
vc = cv.VideoCapture(str(DATA_DIR.joinpath('test.mov')))

# Extract information about the video stream
vid_w, vid_h, fps, tot_frames = get_video_meta(vc)

try:
    # Instantiate an object tracker
    tracker = MvTracker(vid_w, vid_h, max_dist=560, offset_x=50)

    # Track all objects
    prog = tqdm(desc='Frames', total=tot_frames, unit='fps', position=0)
    freqs: dict[int, list] = {}
    pans: dict[int, dict[str, np.ndarray]] = {}
    louds: dict[int, np.ndarray] = {}
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        # Track detected objects
        tracker.track(frame)

        # Go through all the detected objects
        for tid, traj in tracker.trajectories.items():
            # Skip objects with a single point in their trajectory
            if len(traj) < 2:
                prev = np.zeros((2,), dtype=int)
                curr = np.zeros((2,), dtype=int)
                curr_polar = np.zeros((2,), dtype=int)
                icrs = [(0, 0), (0, 0)]
            else:

                # Skip objects with no instantaneous centers of rotations
                if tid not in tracker.icrs:
                    # Replace the instantaneous center of rotations with the detected object's own center
                    icrs = [r.center for r in traj[-fps // 2:]]
                else:

                    # Get the last few instantaneous centers of rotations
                    icrs = tracker.icrs[tid][-fps // 2:]

                # Compute the coordinates of the icr
                # We take the average of a few centers to filter out the jitter
                icr = np.stack(icrs, axis=0).mean(axis=0).astype(int)

                # Translate the coordinates of the last and current position in the icr coordinate system
                prev = cart_to_polar(*(np.array(traj[-2].center, dtype=int) - icr))
                curr = cart_to_polar(*(np.array(traj[-1].center, dtype=int) - icr))

                # Convert the current position into polar coordinates for panning
                curr_polar = cart_to_polar(*traj[-1].center)

            # Compute the angle between the two positions
            theta = np.abs(curr[1] - prev[1])

            # Compute the frequency of movement
            freq = (theta * fps) / (2 * np.pi)

            # Filter frequency to smooth them out
            if tid not in freqs:
                freqs[tid] = [0] * tracker.starts[tid] + [freq]
            else:
                freqs[tid].append(freqs[tid][-1] * (1 - 0.05) + 0.05 * freq)

            # Compute the left, right panning
            left = np.sqrt(2) / 2 * (np.cos(curr_polar[1] + np.pi / 4) - np.sin(curr_polar[1] + np.pi / 4)) * np.ones((int(SAMPLE_RATE/fps), ))
            right = np.sqrt(2) / 2 * (np.cos(curr_polar[1] + np.pi / 4) + np.sin(curr_polar[1] + np.pi / 4)) * np.ones((int(SAMPLE_RATE/fps), ))

            if tid not in pans:
                pans[tid] = {'left': np.concatenate([np.full((int(SAMPLE_RATE * tracker.starts[tid]/fps), ), 0), left], axis=0), 
                             'right': np.concatenate([np.full((int(SAMPLE_RATE * tracker.starts[tid]/fps), ), 0), right], axis=0)}
            else:
                pans[tid]['left'] = np.concatenate([pans[tid]['left'], left], axis=0)
                pans[tid]['right'] = np.concatenate([pans[tid]['right'], right], axis=0)

            # Compute the linear speed of the icr
            loud = np.full((int(SAMPLE_RATE / fps), ), min(500, freqs[tid][-1] * 2 * np.pi * curr[0]))

            # Set loudness based on linear speed of icr
            if tid not in louds:
                louds[tid] = np.concatenate([np.zeros((int(SAMPLE_RATE * tracker.starts[tid]/fps), )), loud], axis=0)
            else:
                louds[tid] = np.concatenate([louds[tid], loud], axis=0)

        # Move the progress bar forward
        prog.update()

    # Find extreme frequencies to normalize values to the [0, 1] interval
    max_freq = 0
    min_freq = np.inf
    for fs in freqs.values():
        max_fs = max(fs)
        min_fs = min(fs)
        if max_fs > max_freq:
            max_freq = max_fs
        if min_fs < min_freq:
            min_freq = min_fs

    song = []
    for tid, fs in freqs.items():
        # Clamp the various frequencies to the closest note
        t_fs = librosa.note_to_hz(librosa.hz_to_note([220 + (440 * (f + 1e-4 - min_freq) / (max_freq - min_freq)) for f in fs])) # 1e-4 has been added to avoid division by zero when converting to note

        # Generate the signal corresponding to the current tracked object
        left, right = [], []
        idx = 0
        while idx < len(t_fs):
            start = idx
            idx += 1
            while idx < len(t_fs) and t_fs[start] == t_fs[idx]:
                idx += 1

            # Generate the tone
            tone = librosa.tone(t_fs[start], sr=SAMPLE_RATE, duration=(idx - start)/fps)

            # Cross fade
            nb_samples = int(SAMPLE_RATE / fps)
            cross_amp = np.linspace(0, 1, num=nb_samples)
            tone[-nb_samples:] *= cross_amp[::-1]  # Fade out the end
            tone[0:nb_samples] *= cross_amp # Fade in the beginning

            # Add the tone to both channels
            left.append(tone)
            right.append(tone)


        # Compile the left and write channels
        left = np.concatenate(left, axis=0)
        right = np.concatenate(right, axis=0)

        # Make sure they are of the right length (rounding errors in tone generation)
        diff = pans[tid]['left'].shape[0] - left.shape[0]
        if diff != 0:
            left = np.concatenate([np.zeros((diff,)), left], axis=0)
            right = np.concatenate([np.zeros((diff,)), right], axis=0)


        # Apply the left/right panning
        left *= pans[tid]['left']
        right *= pans[tid]['right']

        # Apply the loudness to both channels
        #left *= louds[tid] / 100
        #right *= louds[tid] / 100

        # Append the signal to the others
        song.append(np.stack([left, right], axis=1))

    # Mix all the signals together
    song = (np.stack(song, axis=0).mean(axis=0) * np.iinfo(np.int16).max).astype(np.int16)

    # Write the song to file
    wavfile.write('test.wav', SAMPLE_RATE, song)

except KeyboardInterrupt:
    pass
finally:
    # Close the video stream
    vc.release()
