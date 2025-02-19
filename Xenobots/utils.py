#!/usr/bin/env python3
from math import pow

import numpy as np
import cv2 as cv


def get_contours(frame, thres):
    # Convert color to gradients of gray
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Blur image for better edge detection
    frame = cv.blur(frame, (3, 3))

    # Detect edges
    canny = cv.Canny(frame, thres, 2 * thres)

    # Return the contours and associated hierarchy
    return cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


def get_bbox(contour, padding=0):
    # Approximate a closed polygonal curve for the given contour
    poly = cv.approxPolyDP(contour, epsilon=3, closed=True)  # Epsilon is the precision
    # Return the minimal bounding box
    bbox = cv.boundingRect(poly)
    if padding == 0:
        return bbox
    else:
        # Pad the bounding box making sure not to go over the edges
        pad = padding // 2
        return (max(0, bbox[0] - pad), max(0, bbox[1] - pad), bbox[2] + pad, bbox[3] + pad)


def get_rotated_bbox(contour):
    rect = cv.minAreaRect(contour)
    center, size, deg = rect
    rect = cv.RotatedRect(center, size, deg)

    return rect


def get_tl_br(rect):
    return (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3])


class MvTracker(object):
    """Defines an edge detection-based tracker for moving objects."""

    def __init__(self, width, height, max_dist, offset_x=0, offset_y=0, canny_thres=40, nms_thres=0.3):
        """Initializes attributes required for tracking objects.

        Parameters
        ----------
        width: int
            The width of a frame in pixels.

        height: int
            The height of a frame in pixels.

        max_dist: int
            If detected object is further than `max_dist` from the frame center, it will be ignored.

        offset_x: int
            The amount of pixels by which to offset the frame center along the X axis.

        offset_y: int
            The amount of pixels by which to offset the frame center along the Y axis.

        canny_thres: int
            The threshold for the hysteresis procedure part of the Canny edge detector.

        nms_thres: float
            The percentage of overlap between shapes for the non-maximum suppression algorithm to keep only the biggest bounding box.

        """
        super().__init__()

        self._canny_thres = canny_thres
        self._nms_thres = nms_thres
        self._next_id = 0

        self.trajectories: dict[int, list] = {}
        self.icrs: dict[int, list] = {}

        self._frame_center = np.array((offset_x + width // 2, offset_y + height // 2))
        self._max_dist = max_dist

    def track(self, frame):
        """Builds trajectories for detected objects.
        
        Parameters
        ----------
        frame: numpy.ndarray
            A BGR formated video frame.

        """

        
        # Find contours
        contours, hierarchy = get_contours(frame, self._canny_thres)

        # Get the rotated rectangles and associated bounding boxes
        rrects = []
        bboxes = []
        scores = []
        for i, c in enumerate(contours):
            # Find minimum enveloping rotated rectangle
            r_rect = get_rotated_bbox(c)

            # Compute distance to center of frame
            dist = np.linalg.norm(self._frame_center - r_rect.center)

            # Avoid detecting the edges of the petri dish or small foreign bodies
            if dist > self._max_dist or np.any(np.array(r_rect.size) < 5):
                continue

            # Store all information related to the rotated rectangle for later filtering
            rrects.append(r_rect)
            bboxes.append(r_rect.boundingRect())  # In opencv a rectangle is represented by (x, y, w, h)
            scores.append(r_rect.size[0] * r_rect.size[1])

        if len(bboxes) != 0:
            # Non-maximum suppression to keep only non-overlapping rotated rectangles
            indices = cv.dnn.NMSBoxes(np.array(bboxes), np.array(scores), score_threshold=0, nms_threshold=self._nms_thres)
            rrects = [rrects[idx] for idx in indices]

            assigned_d = set()  # Detected objects that have been assigned to a tracked object
            assigned_t = set()  # Tracked objects that have been assigned a detected object
            if len(self.trajectories) > 0:
                tracked_ids = list(self.trajectories.keys())

                for ridx, rect in enumerate(rrects):
                    # Compute the distance between all tracked objects and all detected objects
                    dists = np.full((len(self.trajectories),), np.inf)
                    for tidx, tid in enumerate(tracked_ids):
                        if tid in assigned_t:
                            continue
                        tracked_loc = self.trajectories[tid][-1]  # Get the last known position in the trajectory
                        dists[tidx] = np.linalg.norm((tracked_loc.center[0] - rect.center[0], tracked_loc.center[1] - rect.center[1]))

                    # If there are no tracked objects close, just move on
                    if dists.min() > 30:
                        continue

                    # Assign the detected object to the closest tracked object
                    min_tidx = dists.argmin()
                    tid = tracked_ids[min_tidx]
                    self.trajectories[tid].append(rect)
                    assigned_d.add(ridx)
                    assigned_t.add(tid)

                    # Get the instantaneous center of rotation
                    icr = self.get_icr(tid)
                    if icr is not None:
                        if tid not in self.icrs:
                            self.icrs[tid] = [icr]
                        else:
                            self.icrs[tid].append(icr)

                    if len(assigned_t) == len(tracked_ids):
                        break

            # Treat all remaining rectangles as new objects
            new = set(range(len(rrects))).difference(assigned_d)
            for idx in new:
                self.trajectories[self._next_id] = [rrects[idx]]
                self._next_id += 1

    def get_icr(self, obj_id):
        """Computes the Instantaneous Center of Rotation based on the object's location in three consecutive frames."""

        # If we have less than 3 points, there is nothing that can be done
        trajects = self.trajectories[obj_id]
        #print([r.center for r in trajects[-5:]])
        if len(trajects) < 3:
            return None

        # Get the last three known positions
        x1, y1 = trajects[-1].center
        x2, y2 = trajects[-2].center
        x3, y3 = trajects[-3].center

        # Get the perpendicular bisector for both pairs of points
        try:
            m1 = (x1 - x2) / (y2 - y1)
            b1 = (pow(y2, 2) - pow(y1, 2) + pow(x2, 2) - pow(x1, 2)) / (2 * (y2 - y1))
        except ZeroDivisionError:
            return None

        try:
            m2 = (x2 - x3) / (y3 - y2)
            b2 = (pow(y3, 2) - pow(y2, 2) + pow(x3, 2) - pow(x2, 2)) / (2 * (y3 - y2))
        except ZeroDivisionError:
            return None

        # Compute the point of intersection of the two bisectors
        try:
            x = (b2 - b1) / (m1 - m2)
            y = (m1 * b2 - m2 * b1) / (m1 - m2)

            return (int(x), int(y))
        except ZeroDivisionError:
            return None

