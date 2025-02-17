#!/usr/bin/env python3
from utils import get_contours, get_rotated_bbox
import numpy as np
import cv2 as cv


# Set the threshold for canny edge detection
thres = 40

# Declare a video input
vc = cv.VideoCapture('Data/test.mov')
try:
    # Create a named window to display the results
    cv.namedWindow('Debug', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)

    # Extract information about the video
    vid_w, vid_h = int(vc.get(cv.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv.CAP_PROP_FRAME_HEIGHT))
    # Find the frame center
    vid_c = np.array(( 50 + vid_w // 2, vid_h // 2))
    max_dist = 560
    while True:
        ret, frame = vc.read()
        if not ret:
            break

        # Find contours
        contours, hierarchy = get_contours(frame, thres)

        # Get the rotated rectangles and associated bounding boxes
        rrects = []
        bboxes = []
        scores = []
        for i, c in enumerate(contours):
            # Find minimum enveloping rotated rectangle
            r_rect = get_rotated_bbox(c)

            # Compute distance to center of frame
            dist = np.linalg.norm(vid_c - r_rect.center)

            # Avoid detecting the edges of the petri dish or small foreign bodies
            if dist > max_dist or np.any(np.array(r_rect.size) < 5):
                continue

            # Store all information related to the rotated rectangle for later filtering
            rrects.append(r_rect)
            bboxes.append(r_rect.boundingRect())  # In opencv a rectangle is represented by (x, y, w, h)
            scores.append(r_rect.size[0] * r_rect.size[1])

        if len(bboxes) != 0:
            # Non-maximum suppression to keep only non-overlapping rotated rectangles
            indices = cv.dnn.NMSBoxes(np.array(bboxes), np.array(scores), score_threshold=0, nms_threshold=0.5)

            # TODO: Since we are already tracking all the xenobots, we might as well ditch the tracker and build the trajectory ourself
            # TODO: Dead Reckoning based on position, distance and speed might work well enough
            # TODO: Maybe even just optical flow?

            for idx in indices:
                # Draw the rotated rectangles
                box = np.intp(cv.boxPoints(rrects[idx]))
                cv.drawContours(frame, [box], 0, (0, 255, 0))


        cv.imshow('Debug', frame)
        cv.pollKey()
except KeyboardInterrupt:
    pass
finally:
    # Close the video stream
    vc.release()

    # Close all windows
    cv.destroyAllWindows()
