#!/usr/bin/env python3
import numpy as np
import cv2 as cv
from tqdm import tqdm

from utils import MvTracker, get_video_meta


colors = [
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
# Set the threshold for canny edge detection
thres = 40

win_name = 'Debug'

# Declare a video input
vc = cv.VideoCapture('Data/test.mov')
try:
    # Create a named window to display the results
    cv.namedWindow(win_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)

    # Extract information about the video
    vid_w, vid_h, fps, tot_frames = get_video_meta(vc)

    # Instantiate a movement tracker
    tracker = MvTracker(vid_w, vid_h, max_dist=560, offset_x=50)
    prog = tqdm(desc='Frames', total=tot_frames, unit='fps', position=0)
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        # Track objects
        tracker.track(frame)

        for tid, tracks in tracker.trajectories.items():
            # Draw whole trajectory
            for idx in range(1, len(tracks)):
                cv.line(frame, np.intp(tracks[idx - 1].center), np.intp(tracks[idx].center), colors[tid%len(colors)][::-1], 2)

            # Draw Instantaneous Center of Rotation
            try:
                icrs = tracker.icrs[tid][-fps // 2:]
                icr = np.stack(icrs, axis=0).mean(axis=0).astype(int)
                frame = cv.circle(frame, icr, 1, (0, 0, 255), 3)
            except KeyError:
                pass

        cv.imshow(win_name, frame)
        cv.pollKey()
        prog.update()
except KeyboardInterrupt:
    pass
finally:
    # Close the video stream
    vc.release()

    # Close all windows
    cv.destroyAllWindows()
