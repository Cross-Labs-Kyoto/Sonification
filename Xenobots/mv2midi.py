#!/usr/bin/env python3
import cv2 as cv
from mido import MidiFile, MetaMessage, Message, second2tick
import librosa

from utils import mv_to_freqs_n_pans, get_video_meta
from settings import DATA_DIR


# String instruments only
#INSTRUMENTS = list(range(40, 56))
INSTRUMENTS = [
    27,
    41,
    58,
    65,
    73,
    80,
    16,
    33
]


if __name__ == "__main__":
    # Define the path to the video file
    video_path = DATA_DIR.joinpath('test.mov')

    # Instantiate a video reader
    vc = cv.VideoCapture(str(video_path))

    # Extract information about the video stream
    _, _, fps, _ = get_video_meta(vc)

    try:
        # Get the frequencies and pannings for all tracked objects
        freqs, pans = mv_to_freqs_n_pans(vc, decay_rate=0.025)
    except KeyboardInterrupt:
        pass
    finally:
        # Close the video stream
        vc.release()

    # Open a new MIDI file
    midi_file = MidiFile(ticks_per_beat=480)  # Defaults are: ticks per beat: 480, tempo: 500000 micro sec/quarter note, time signature: 4/4 => one quarter per beat

    # Get the number ticks between frames
    d_t = second2tick(1/fps, ticks_per_beat=480, tempo=500000)

    # Build the tracks for each object
    for tid, fs in freqs.items():
        # Add a new track to the file for the current object
        track = midi_file.add_track(name=str(tid))

        # Set the tempo and time signature
        track.append(MetaMessage('set_tempo', tempo=500000))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4))


        # Set the instrument for the current object
        track.append(Message('program_change', channel=tid, program=INSTRUMENTS[tid % len(INSTRUMENTS)]))
        track.append(Message('program_change', channel=tid+1, program=INSTRUMENTS[tid % len(INSTRUMENTS)]))


        # Translate the frequencies to notes
        notes = [round(librosa.hz_to_midi(220 + f * 440)) for f in fs]

        # Send midi messages according to notes and pans
        prev_note = None
        for note, pan_l, pan_r in zip(notes, pans[tid]['left'], pans[tid]['right']):
            # Start a new note
            if prev_note != note:
                if prev_note is not None:
                    # Stop the previous note
                    track.append(Message('note_off', channel=tid, note=prev_note, velocity=64, time=d_t))
                    track.append(Message('note_off', channel=tid+1, note=prev_note, velocity=64, time=0))

                    # Start a new note
                    track.append(Message('note_on', channel=tid, note=note, velocity=round(pan_l * 63 + 64), time=0))
                    track.append(Message('note_on', channel=tid+1, note=note, velocity=round(pan_r * 63 + 64), time=0))
                else:
                    # Start a new note
                    track.append(Message('note_on', channel=tid, note=note, velocity=round(pan_l * 63 + 64), time=d_t))
                    track.append(Message('note_on', channel=tid+1, note=note, velocity=round(pan_r * 63 + 64), time=0))

                # Store the new note
                prev_note = note

            # Modulate the current note using the left/right pan
            else:
                # TODO: Is aftertouch better here?
                track.append(Message('polytouch', channel=tid, note=note, value=round(pan_l * 63 + 64), time=d_t))
                track.append(Message('polytouch', channel=tid+1, note=note, value=round(pan_r * 63 + 64), time=0))

        # Make sure the last note ends
        track.append(Message('note_off', channel=tid, note=notes[-1], velocity=64, time=0))
        track.append(Message('note_off', channel=tid+1, note=notes[-1], velocity=64, time=0))

    # Save the midi file to disk
    midi_file.save(filename=DATA_DIR.joinpath('test.midi'))
    print(f'Estimated duration: {midi_file.length} sec')
