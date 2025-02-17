#!/usr/bin/env python3
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
