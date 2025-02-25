# Description
The goal of this project is to use sound as an additional modality to recognize patterns in the movement and calcium activity of Xenobots.
Taking for granted that humans are very capable of recognizing patterns in sequences of audio segment, we investigate interesting mappings between the trajectories observed from xenobots, or their calcium activity, and different properties of sound.

So far, the focus has been on xenobot movement. However, before translating trajectories into sound, we had to process the video stream to detect the various agents, then track their movement across frames.
The detection heavily relies on [OpenCV's Canny edge detector](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga2a671611e104c093843d7b7fc46d24af) which extracts contours from each video frame. Bounding boxes are then fitted around those contours, and a [Non-Maximum Suppression](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html#ga9d118d70a1659af729d01b10233213ee) provides us with actual detected objects.
To track objects and build trajectories, we apply a naive algorithm which associates bounding boxes in two consecutive frames based on their euclidean distance.

Since most of the Xenobots seem to follow a cork-screw pattern of motion, we decided to also track their [Instant Center of Rotation (ICR)](https://www.wikiwand.com/en/articles/Instant_centre_of_rotation). Given the location of a xenobot, and its ICR, we established the following mapping between movement and sound:
+ The location of the object's center on the X axis dictates the left/right panning.
+ Frequency of rotation around the ICR determines the pitch of the associated tone.
+ For the moment, loudness is tied to speed of movement. However, since differentiating between different volumes is quite hard, we will be experimenting with assigning each tracked object its own instrument.

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
At the moment, the path to the video used for sonification of the Xenobots' trajectories has been hardcoded in the `mv2snd.py` script (l.13). Therefore, it is assumed that both the `Data` directory, and the `test.mov` file within it exist. A simple check is made to test for the file's existence.
The video used as a test for the `mv2snd.py` script is a copy of the `Xenobot baseline movement - single and swarm/2025-01-22 - D7 Bot swarm 1 .MOV` file that was provided to us.
After creating the `Data` folder:
```bash
$> cd Sonification/Xenobots/
$> mkdir Data
```
Copy the test video:
```bash
$> cp Xenobot baseline movement - single and swarm/2025-01-22 - D7 Bot swarm 1 .MOV Data/test.mov
```
To process the video and transform the detected trajectories into sound, run the `mv2snd.py` script as follow:
```bash
$> cd Sonification/Xenobots/
$> poetry shell
$> python mv2snd.py
```
This will produce a `test.wav` audio file in the current working directory. The produced sound should be of the same length as the provided video, such that you can either use third-party software (e.g.: ffmpeg) to combine the audio and video, or open both files separately.

# Feedback / Error
To provide any feedback or if you encounter problems using this repository, please open a new [**Issue**](https://github.com/Cross-Labs-Kyoto/Sonification/issues) and assign it the tag corresponding to the particular sub-project you are interested in.
