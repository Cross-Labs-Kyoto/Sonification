#!/usr/bin/env python3
import cv2 as cv
from tqdm import tqdm
import numpy as np

from utils import MvTracker, get_video_meta, VideoIterator


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
with VideoIterator('Data/test.mov') as vi:
    # Create a named window to display the results
    cv.namedWindow(win_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)

    # Extract information about the video
    vid_w, vid_h, fps, tot_frames = get_video_meta(vi)

    # Instantiate a movement tracker
    tracker = MvTracker(vid_w, vid_h, max_dist=560, offset_x=50)
    prog = tqdm(desc='Frames', total=tot_frames, unit='fps', position=0)
    for frame in vi:
        # Track objects
        tracker.track(frame)
        for obj in tracker.tracked_objects:
            data = obj.last_detection.data
            r_rect = data['rrect']
            bbox = data['bbox']
            x1, y1 = bbox[0:2]
            x2, y2 = bbox[0:2] + bbox[2:]

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(vid_w, x2), min(vid_h, y2)
            # Extract the xenobot
            xeno = frame[y1:y2, x1:x2, :]

            # Rotate inside of bbox providing size from the rotated rectangle
            # TODO: See here for rotation: https://docs.opencv.org/4.12.0/da/d6e/tutorial_py_geometric_transformations.html#autotoc_md1365
            # TODO: RotatedRectangle.size returns a tuple of (width, height)
            # TODO: RotatedRectangle.center returns a tuple of (x, y)
            # TODO: RotatedRectangle.angle returns a angle in degrees
            bbox_center = bbox[:2] + bbox[:2] / 2
            rot = cv.getRotationMatrix2D(bbox_center, 90 - r_rect.angle, 1)
            xeno = cv.warpAffine(xeno, rot, np.asarray(r_rect.size, dtype=int))
            cv.imshow(f'xeno{obj.id}', xeno)

            box = cv.boxPoints(r_rect).astype(int)
            cv.drawContours(frame, [box], 0, colors[obj.id%len(colors)][::-1], 2)

        cv.imshow(win_name, frame)
        cv.pollKey()
        prog.update()

# Close all windows
cv.destroyAllWindows()
