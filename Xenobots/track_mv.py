#!/usr/bin/env python3
import cv2 as cv
from tqdm import tqdm

from utils import MvTracker, VideoIterator


DEBUG = True
# Set the threshold for canny edge detection
thres = 40

# Declare a video input
with VideoIterator('Data/test.mov') as vi:

    # Extract information about the video
    vid_w, vid_h, fps, tot_frames = vi.get_metadata()

    if DEBUG:
        # Initialize video writer used for debugging
        vw = cv.VideoWriter(str('Data/dbg.mp4'), cv.VideoWriter_fourcc(*'mp4v'), fps, (vid_w, vid_h), True)  # True indicate that the video is in color

    try:
        # Instantiate a movement tracker
        tracker = MvTracker(vid_w, vid_h, debug=DEBUG)
        prog = tqdm(desc='Frames', total=tot_frames, unit='fps', position=0)
        for frame in vi:
            # Track objects
            tracker.track(frame)

            if DEBUG:
                # Write debug video
                vw.write(frame)

            # Increase progress bar
            prog.update()
    except KeyboardInterrupt:
        pass
    finally:
        if DEBUG:
            # Close video file
            vw.release()
