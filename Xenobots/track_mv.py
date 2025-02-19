#!/usr/bin/env python3
import numpy as np
import cv2 as cv

from utils import MvTracker


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

# Declare a video input
vc = cv.VideoCapture('Data/test.mov')
try:
    # Create a named window to display the results
    cv.namedWindow('Debug', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)

    # Extract information about the video
    vid_w, vid_h = int(vc.get(cv.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(vc.get(cv.CAP_PROP_FPS))
    # Find the frame center
    vid_c = np.array(( 50 + vid_w // 2, vid_h // 2))
    max_dist = 560

    # Instantiate a movement tracker
    tracker = MvTracker(vid_w, vid_h, max_dist, offset_x=50)
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
                icr = np.stack(tracker.icrs[tid][-fps // 2:], axis=0).mean(axis=0).astype(int)
                frame = cv.circle(frame, icr, 1, (0, 0, 255), 3)
            except KeyError:
                pass

        cv.imshow('Debug', frame)
        cv.pollKey()
except KeyboardInterrupt:
    pass
finally:
    # Close the video stream
    vc.release()

    # Close all windows
    cv.destroyAllWindows()
