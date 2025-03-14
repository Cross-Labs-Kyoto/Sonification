# Description
The goal of this project is to use sound as an additional modality to recognize patterns in the movement and calcium activity of Xenobots.
Taking for granted that humans are very capable of recognizing patterns in sequences of audio segment, we investigate interesting mappings between the trajectories observed from xenobots, or their calcium activity, and different properties of sound.

## Movement
Before translating trajectories into sound, we had to process the video stream to detect the various agents, then track their movement across frames.
The detection heavily relies on [OpenCV's Canny edge detector](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga2a671611e104c093843d7b7fc46d24af) which extracts contours from each video frame. Bounding boxes are then fitted around those contours, and a [Non-Maximum Suppression](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html#ga9d118d70a1659af729d01b10233213ee) provides us with actual detected objects.
To track objects and build trajectories, we apply a naive algorithm which associates bounding boxes in two consecutive frames based on their euclidean distance.

Since most of the Xenobots seem to follow a cork-screw pattern of motion, we decided to also track their [Instant Center of Rotation (ICR)](https://www.wikiwand.com/en/articles/Instant_centre_of_rotation). Given the location of a xenobot, and its ICR, we established the following mapping between movement and sound:
+ The location of the object's center on the X axis dictates the left/right panning.
+ Frequency of rotation around the ICR determines the pitch of the associated tone.
+ For the moment, loudness is tied to speed of movement. However, since differentiating between different volumes is quite hard, we will be experimenting with assigning each tracked object its own instrument.

## Calcium activity
To translate calcium activity into sound, the pre-processing stages are a bit different. In this case, it is not possible to use a standard edge detection algorithm. Instead, we computed the perceived brightness of each pixel in a frame based on its RGB value (see [this post](https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color#answer-56678483) for the formula used). A non-zero thresholding filter was then applied to ignore any pixel bellow a certain brightness. Here the threshold is chosen as `mean + 2 * std`, where `mean` and `std` are the average brightness computed over the whole frame, and the standard deviation, respectively. The resulting grayscale frame is then passed to the same bounding box finding and non-maximum suppression pipeline as above.

Mapping between calcium activity and sound is a bit more straight forward. A grid of 8x8 was overlayed on top of the frame. For each cell, we computed the cumulative brightness of all bounding boxes that were contained inside.
Brightness was then translated into both the duration of a note, and its loudness. Therefore the brighter a cell is the longer and louder the corresponding note. A decay rate of 0.4 was applied to the notes' duration.
Instead of increasing the note's frequency and panning left/right based on the cell's position, we used a [Hilbert curve](https://www.wikiwand.com/en/articles/Hilbert_curve) (see [here](https://github.com/galtay/hilbertcurve) for the implement used) to translate position into note. Using a Hilbert curve means that cells that are close to one another in space with have similar notes. Consequently, activity in the lower left corner of the grid correspond to low frequency notes, while the lower right corner is associated with high frequency ones.

# Install
This entire sub-project uses python 3.12.8 as its main programming language. It also relies on [Poetry](https://python-poetry.org/docs/) for managing its dependencies.
Therefore, the recommended way of installing this project is as follow:
1. Download and install [python 3.12.8](https://www.python.org/downloads/release/python-3128/) or use a virtual environment (e.g.: [PyEnv](https://github.com/pyenv/pyenv)),
2. Follow [Poetry's documentation](https://python-poetry.org/docs/#installing-with-the-official-installer) to get it setup on,
3. Clone this repository: `$> git clone https://github.com/Cross-Labs-Kyoto/Sonification.git`
4. Navigate to the `Xenobots` sub-directory: `$> cd Sonification/Xenobots/`
5. Install all dependencies using poetry: `$> poetry install`
6. If everything worked correctly, you should now be able to use the virtual environment using the following command: `$> poetry shell`
Note that before executing any `python` related commands in this sub-directory you will need to activate the virtual environment.

# Usage
## Movement
The `mv2snd.py` script translates the trajectory of one or more xenobots directly into a `.wav` file. To do so simply type:
```bash
$> cd Sonification/Xenobots/
$> poetry shell
$> python mv2snd.py relative/path/to/video/file.mov
```
It is possible to adjust the threshold for the Canny edge detector by providing an integer value to the `-t` parameter (e.g.: `python mv2snd.py -t 120`). The resulting sound file is placed in the same directory, with the same name, as the input video file. The produced sound should be of the same length as the provided video, such that you can either use third-party software (e.g.: ffmpeg) to combine the audio and video, or open both files separately.

## Calcium activity
The `act2midi.py` scripts translates the calcium activity of a xenobot into a [MIDI](https://www.wikiwand.com/en/articles/MIDI) file. On the contrary to the previous program, though, it works on a directory, and iterates through all files. The path has been hard-coded into the python file itself (line 30). To run the program use the following commands:
```bash
$> cd Sonification/Xenobots/
$> poetry shell
$> python act2midi.py
```
This will output a MIDI file with the same name and in the same location as the input video.
MIDI files can then be converted into audio files using a third-party software, such as [Timidity++](https://timidity.sourceforge.net/), or played directly using a synthesizer like [FluidSynth](https://www.fluidsynth.org/). The produced sound should be of the same length as the provided video, such that you can either use third-party software (e.g.: ffmpeg) to combine the audio and video, or open both files separately.

# Feedback / Error
To provide any feedback or if you encounter problems using this repository, please open a new [**Issue**](https://github.com/Cross-Labs-Kyoto/Sonification/issues) and assign it the tag corresponding to the particular sub-project you are interested in.
