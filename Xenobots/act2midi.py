#!/usr/bin/env python3
from collections import defaultdict

import numpy as np
import cv2 as cv
from hilbertcurve.hilbertcurve import HilbertCurve
from mido import MidiFile, MetaMessage, Message, second2tick

from utils import get_video_meta, get_bbox
from settings import DATA_DIR


def get_brightness(frame):
    # See here for reference and formula: https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color#answer-56678483
    # Standardize the frame
    std_frame = frame / 255.
    # Gamma encode the frame
    gRGB_frame = np.where(std_frame <= 0.04045, std_frame / 12.92, np.pow((std_frame + 0.055) / 1.055, 2.4))
    # Compute luminance
    lum = np.sum(gRGB_frame * np.array([0.2126, 0.7152, 0.0722]), axis=2)
    # Compute perceived brightness (white = 1, black = 0)
    return np.where(lum <= 216/24389, lum * (24389 / 27), np.pow(lum, 1/3) * 116 - 16) / 100


if __name__ == "__main__":
    # Compute the 3rd iteration for the 2D Hilbert Curve
    hb = HilbertCurve(3, 2)

    #for vid_path in chain(DATA_DIR.joinpath('Xenobot baseline Calcium videos').iterdir(), DATA_DIR.joinpath('Xenobot post-stimulus Calcium videos').iterdir()):
    for vid_path in DATA_DIR.joinpath('Xenobot post-stimulus Calcium videos').iterdir():
        # Ignore anything that is not a file
        if not vid_path.is_file() or vid_path.suffix == '.midi':
            continue

        # Open video stream
        vc = cv.VideoCapture(str(vid_path))

        # Extract video's meta information
        width, height, fps, tot_frames = get_video_meta(vc)

        try:
            # Process all frames
            cnt = 0
            cell_brights = []
            min_bright = np.inf
            max_bright = 0
            while True:
                # Get the next frame
                ret, frame = vc.read()
                if not ret:
                    break

                # Remove the clock in the top right corner
                frame[5:30, 335:510,:] = 0

                # Remove the scale in the bottom right corner
                # TODO: Remove for non-scaled videos
                frame[height-45:, width-150:,:] = 0
                frame[height-10:, width-300:,:] = 0

                if cnt % 3 == 0:
                    # Convert frame to rgb format
                    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                    # Compute the brightness of all pixels
                    frame_bright = get_brightness(rgb_frame)

                    # Threshold the frame to highlight calcium activity
                    mean = np.mean(frame_bright)
                    std = np.std(frame_bright)
                    _, frame_bright = cv.threshold(frame_bright, mean + 2 * std, 1, cv.THRESH_TOZERO)

                    # Cast the brightness to grayscale
                    grayscale_bright = (frame_bright * 255).astype(np.uint8)

                    # Find the contours
                    contours, hierarchy = cv.findContours(grayscale_bright, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                    # Get bounding box and compute its score (=area)
                    bboxes = []
                    scores = []
                    for ctn in contours:
                        bb = get_bbox(ctn)
                        bboxes.append(bb)
                        scores.append(bb[2] * bb[3])

                    # Non-maximum suppression
                    idxs = cv.dnn.NMSBoxes(np.array(bboxes), np.array(scores, dtype=float), score_threshold=50, nms_threshold=0.1)
                    brights = defaultdict(int)
                    for idx in idxs:
                        # Extract the box's definition
                        bx, by, bw, bh = bboxes[idx]
                        # Compute the box's brightness
                        bb = frame_bright[by:by+bh, bx:bx+bw].sum().item()
                        # Keep track of cell brightness
                        cell_x, cell_y = bx // 64, by // 64
                        cell = hb.distance_from_point((cell_x, cell_y))
                        brights[cell] += bb

                    # Keep track of the minimum and maximum cell brightness
                    try:
                        max_b = max(brights.values())
                    except ValueError:
                        max_b = 0
                    try:
                        min_b = min(brights.values())
                    except ValueError:
                        min_b = np.inf

                    if max_b > max_bright:
                        max_bright = max_b
                    if min_b < min_bright:
                        min_bright = min_b

                    # Keep track of the cells' brightness
                    cell_brights.append(brights)

                # Increase the frame counter
                cnt += 1

        except KeyboardInterrupt:
            continue
        finally:
            # Close the video stream
            vc.release()

        # Open a new MIDI file
        midi_file = MidiFile(ticks_per_beat=480)  # Defaults are: ticks per beat: 480, tempo: 500000 micro sec/quarter note, time signature: 4/4 => one quarter per beat

        # Add a new track to the file for the current object
        track = midi_file.add_track(name='main')

        # Set the tempo and time signature
        track.append(MetaMessage('set_tempo', tempo=500000))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4))


        # Set the instrument
        track.append(Message('program_change', channel=1, program=40))

        # Process the sequence of brightnesses
        start_note = 32
        decay_rate = 0.4
        playing = set()
        for idx, brights in enumerate(cell_brights):
            # Get the number ticks between frames
            d_t = 3 * second2tick(1/fps, ticks_per_beat=480, tempo=500000)
            
            # Go through the cells
            for cell in range(64):
                # Get the corresponding note
                note = cell + start_note

                # Play new notes
                if note not in playing:
                    if brights[cell] == 0:
                        continue

                    bright = (brights[cell] - min_bright) / (max_bright - min_bright)
                    track.append(Message('note_on', channel=1, note=note, velocity=round(127 * bright), time=d_t))
                    d_t -= d_t
                    playing.add(note)
                else:
                    # Compute the decayed brightness
                    bright = cell_brights[idx-1][cell] * (1 - decay_rate) + decay_rate * (brights[cell] - min_bright) / (max_bright - min_bright)

                    # Ignore note that are too quiet
                    if bright <= 0:
                        # If the note was previously played, stop it
                        track.append(Message('note_off', channel=1, note=note, velocity=64, time=d_t))
                        playing.remove(note)
                        d_t -= d_t

                        # Set the cell to 0 for later computation of the duration
                        bright = 0
                    else:
                        # Modulate the corresponding note
                        track.append(Message('polytouch', channel=1, note=note, value=round(127 * bright), time=d_t))
                        d_t -= d_t

                # Keep track of the decayed cell value
                brights[cell] = bright

        # Stop any still playing note
        d_t = 3 * second2tick(1/fps, ticks_per_beat=480, tempo=500000)
        for note in playing:
            track.append(Message('note_off', channel=1, note=note, velocity=32, time=d_t))
            d_t -= d_t

        # Save the midi file to disk
        midi_file.save(filename=vid_path.with_suffix('.midi'))
