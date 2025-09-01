#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from collections import deque

import numpy as np
import cv2 as cv
from tqdm import tqdm

from utils import MvTracker, VideoIterator
from settings import COLORS


def track_mv(vid_it, vid_w, vid_h, tot_frames):
    prog = tqdm(desc='Frames', total=tot_frames, position=0)
    # Instantiate a movement tracker
    tracker = MvTracker(vid_w, vid_h)
    for frame in vid_it:
        prog.update()
        # Track objects
        tracker.track(frame)

    return tracker.tracks


def track_to_video(trk, f_path, vid_w, vid_h, fps, tot_frames):
    # Ignore anomalously short tracks
    if len(trk.contours) < tot_frames / 2:
        return 

    # Initialize a video writer for the track
    vw = cv.VideoWriter(str(itm.with_stem(f'{f_path.stem}_{trk.id}').with_suffix('.mp4')),
                        cv.VideoWriter_fourcc(*'mp4v'), fps, (vid_w, vid_h), True)  # True indicate that the video is in color
    try:
        # Get new frame
        frame = np.zeros((vid_h, vid_w, 3), dtype=np.float32)
        color = COLORS[trk.id % len(COLORS)]
        for cntr in trk.contours:
            # Draw contour
            frame = cv.drawContours(frame, np.expand_dims(cntr, axis=0), -1, color)

            # Write frame
            vw.write(frame.astype(np.uint8))

            # Decay frame to get a trail
            frame *= 0.8

    except KeyboardInterrupt:
        pass
    finally:
        # Close video file
        vw.release()


def track_to_midi(trk, f_path, tot_frames):
    # Ignore anomalously short tracks
    if len(trk.contours) < tot_frames / 2:
        return 

    # TODO: Transform the object tracklet into a midi track
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-d', '--dir', type=Path, dest='dir', required=True,
                        help="The relative path to the directory containing the videos to process.")

    # Parse arguments
    args = parser.parse_args()
    if not (args.dir.exists() and args.dir.is_dir()):
        print(f'Error: The provided path either does not exist or is not a directory: {args.dir}')


    # Iterate through the directory
    queue = deque([args.dir.expanduser().resolve()])
    while True:
        try:
            root = queue.pop()
        except IndexError:
            break

        for itm in root.iterdir():
            # Put sub-directories in the queue
            if itm.is_dir():
                print(f'Adding {itm} to the queue')
                queue.append(itm)
                # And move on to next item
                continue

            with VideoIterator(itm) as vi:
                # Extract information about the video
                vid_w, vid_h, fps, tot_frames = vi.get_metadata()

                # Track movement and store tracks for later processing
                print(f'Processing: {itm.name}')
                tracks = track_mv(vi, vid_w, vid_h, tot_frames)

            # For all tracks
            prog = tqdm(desc='Tracks', total=len(tracks), position=0)
            for trk in tracks.values():
                # Increase progress bar
                prog.update()
                # Translate tracks into videos
                track_to_video(trk, itm, vid_w, vid_h, fps, tot_frames)

