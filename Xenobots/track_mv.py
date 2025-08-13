#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
from collections import deque
import numpy as np
import cv2 as cv
from tqdm import tqdm

from utils import MvTracker, VideoIterator


COLORS = [
    (40, 42, 54),
    (248, 248, 242),
    (139, 233, 253),
    (80, 250, 123),
    (255, 184, 108),
    (255, 121, 198),
    (68, 71, 90),
    (189, 147, 249),
    (255, 85, 85),
    (241, 250, 140),
    (98, 114, 164)
]

parser = ArgumentParser()
parser.add_argument('-d', '--dir', type=Path, dest='dir', help="The relative path to the directory containing the videos to process.")

if __name__ == "__main__":
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

            # Process video files
            print(f'Processing: {itm}')
            with VideoIterator(itm) as vi:
                # Extract information about the video
                vid_w, vid_h, fps, tot_frames = vi.get_metadata()

                # Instantiate a movement tracker
                tracker = MvTracker(vid_w, vid_h)
                prog = tqdm(desc='Frames', total=tot_frames, unit='fps', position=0)
                for frame in vi:
                    # Track objects
                    tracker.track(frame)

                    # Increase progress bar
                    prog.update()

                prog = tqdm(desc='Tracks', total=len(tracker.tracks), position=0)
                for trk in tracker.tracks.values():
                    # Increase progress bar
                    prog.update()

                    # Ignore anomalously short tracks
                    if len(trk.contours) < tot_frames / 2:
                        continue

                    # Initialize a video writer for the track
                    vw = cv.VideoWriter(str(itm.with_stem(f'{itm.stem}_{trk.id}').with_suffix('.mp4')),
                                        cv.VideoWriter_fourcc(*'mp4v'), fps, (vid_w, vid_h), True)  # True indicate that the video is in color
                    try:
                        # Get new frame
                        frame = np.zeros((vid_h, vid_w, 3), dtype=np.float32)
                        color = COLORS[trk.id % len(COLORS)]
                        for cntr in trk.contours:
                            # Draw contour
                            frame = cv.drawContours(frame, np.expand_dims(cntr, axis=0), -1, color)

                            # Write debug video
                            vw.write(frame.astype(np.uint8))

                            # Decay frame to get a trail
                            frame *= 0.8

                    except KeyboardInterrupt:
                        pass
                    finally:
                        # Close video file
                        vw.release()
