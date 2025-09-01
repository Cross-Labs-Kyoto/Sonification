#!/usr/bin/env python3
import hashlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
from norfair import Detection, Tracker, OptimizedKalmanFilterFactory
from tqdm import tqdm
from loguru import logger


def get_contours(frame, thres):
    # Convert color to gradients of gray
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Blur image for better edge detection
    frame = cv.blur(frame, (3, 3))

    # Detect edges
    canny = cv.Canny(frame, thres, 2 * thres)

    # Return the contours and associated hierarchy
    return cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


def get_rotated_bbox(contour):
    rect = cv.minAreaRect(contour)
    center, size, deg = rect
    rect = cv.RotatedRect(center, size, deg)

    return rect

def arctan(y,x):
    # Return the angle in radians in the range [0, 2pi], starting at (0, 1) and going counter-clockwise
    return np.arctan2(y, x) + np.abs(np.sign(np.arctan2(y, x)) - 1) * np.pi

def cart_to_polar(x, y):
    return np.linalg.norm((x, y)), arctan(y, x)


def polar_to_cart(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def get_tl_br(rect):
    return (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3])


def mv_to_freqs_n_pans(video_capture, decay_rate=0.05):
    # Extract information about the video stream
    vid_w, vid_h, fps, tot_frames = video_capture.get_video_meta()

    # Instantiate an object tracker
    tracker = MvTracker(vid_w, vid_h, max_dist=560, offset_x=50)

    # Track all objects
    prog = tqdm(desc='Frames', total=tot_frames, unit='frame', position=0)
    freqs: dict[int, list] = {}
    pans: dict[int, dict[str, list]] = {}
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Track detected objects
        tracker.track(frame)

        # Go through all the detected objects
        for tid, traj in tracker.trajectories.items():
            # Handle objects with a single point in their trajectory
            if len(traj) < 2:
                prev = np.zeros((2,), dtype=int)
                curr = np.zeros((2,), dtype=int)
                curr_polar = np.zeros((2,), dtype=int)
                icrs = [(0, 0), (0, 0)]
            else:

                # Convert the current position into polar coordinates for panning
                curr_polar = cart_to_polar(traj[-1].center[0] - (tracker.frame_center[0]), tracker.frame_center[1] - traj[-1].center[1])

                # Handle objects with no instantaneous centers of rotations
                if tid not in tracker.icrs:
                    # Replace the instantaneous center of rotations with the detected object's own center
                    icrs = [r.center for r in traj[-fps // 2:]]
                else:

                    # Get the last few instantaneous centers of rotations
                    icrs = tracker.icrs[tid][-fps // 2:]

                # Compute the coordinates of the icr
                # We take the average of a few centers to filter out the jitter
                icr = np.stack(icrs, axis=0).mean(axis=0).astype(int)

                # Translate the coordinates of the last and current position in the icr coordinate system
                prev = cart_to_polar(*(np.array(traj[-2].center, dtype=int) - icr))
                curr = cart_to_polar(*(np.array(traj[-1].center, dtype=int) - icr))

            # Compute the angle between the two positions
            theta = np.abs(curr[1] - prev[1])

            # Compute the frequency of movement
            freq = (theta * fps) / (2 * np.pi)

            # Filter frequency to smooth them out
            if tid not in freqs:
                freqs[tid] = [0] * tracker.starts[tid] + [freq]
            else:
                freqs[tid].append(freqs[tid][-1] * (1 - decay_rate) + decay_rate * freq)

            # Compute the left, right panning
            left = -(np.cos(np.abs(curr_polar[1]))).item()
            right = (np.cos(np.abs(curr_polar[1]))).item()

            if tid not in pans:
                pans[tid] = {'left': [0] * tracker.starts[tid] + [left],
                             'right': [0] * tracker.starts[tid] + [right]}
            else:
                pans[tid]['left'].append(left)
                pans[tid]['right'].append(right)

        # Move the progress bar forward
        prog.update()

    # Normalize the frequencies to the interval [0, 1]
    max_freq = 0
    min_freq = np.inf
    for fs in freqs.values():
        max_fs = max(fs)
        min_fs = min(fs)
        if max_fs > max_freq:
            max_freq = max_fs
        if min_fs < min_freq:
            min_freq = min_fs

    for tid in freqs.keys():
        freqs[tid] = [(f - min_freq + 1e-8) / (max_freq - min_freq) for f in freqs[tid]]  # 1e-8 has been added to avoid division by zero when converting to note

    # Return the frequencies and panning for all tracked objects
    return freqs, pans


class MvTracker(object):
    """Defines an edge detection-based tracker for moving objects."""

    def __init__(self, width, height, canny_thres=40, nms_thres=0.3, debug=False):
        """Initializes attributes required for tracking objects.

        Parameters
        ----------
        width: int
            The width of a frame in pixels.

        height: int
            The height of a frame in pixels.

        canny_thres: int
            The threshold for the hysteresis procedure part of the Canny edge detector.

        nms_thres: float
            The percentage of overlap between shapes for the non-maximum suppression algorithm to keep only the biggest bounding box.

        debug: bool
            A flag indicating whether to execute the tracker in debug mode or not.

        """
        super().__init__()

        self._canny_thres = canny_thres
        self._nms_thres = nms_thres
        self.tracks = {}

        # TODO: Adapt distance_threshold, if too many dropped objects
        self._tracker = Tracker(distance_function='euclidean', distance_threshold=40,
                                hit_counter_max=5,
                                filter_factory=OptimizedKalmanFilterFactory(R=0.1, Q=4))
                                #reid_distance_function=lambda x, y: cdist(x.estimate, y.estimate, metric='euclidean'),
                                #reid_distance_threshold=50)

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
        bboxes = []
        scores = []
        cntrs = []  # This is necessary to discard small bodies and edge of the petri dish
        for i, c in enumerate(contours):
            # Find minimum enveloping rotated rectangle
            r_rect = get_rotated_bbox(c)

            # Avoid detecting the edges of the petri dish or small foreign bodies
            if np.any(np.array(r_rect.size) < 10) or np.any(np.array(r_rect.size) > 75):
                continue

            # Store all information related to the rotated rectangle for later filtering
            bboxes.append(r_rect.boundingRect())  # In opencv a rectangle is represented by (x, y, w, h)
            scores.append(r_rect.size[0] * r_rect.size[1])
            cntrs.append(c)

        if len(bboxes) != 0:
            # Turn scores and bboxes into numpy arrays for ease of manipulation
            scores = np.array(scores)
            bboxes = np.array(bboxes)

            # Non-maximum suppression to keep only non-overlapping bboxes
            indices = cv.dnn.NMSBoxes(bboxes, scores, score_threshold=0, nms_threshold=self._nms_thres)

            if len(indices) > 10:
                logger.warning(f'Detected an awful lot of objects ({len(indices)})!!! Increase the canny threshold for better detection.')

            # Keep only non-overlapping bboxes
            bboxes = bboxes[indices]
            # And corresponding contours
            contours = [cntrs[idx] for idx in indices]

            # Find the center of every bboxes
            centers = bboxes[:, :2] + bboxes[:, 2:] / 2

            # Need to add a dimension to fit NorFair input format
            centers = np.expand_dims(centers, axis=1)

            # Declare the relevant detections
            detections = [Detection(center, data=idx) for idx, center in enumerate(centers)]

            # Update tracker
            tracked_objs = self._tracker.update(detections)

            # Update tracks
            for obj in tracked_objs:
                # Ignore object that are initializing or stale
                if obj.id is None and not obj.live_points[0]:
                    continue

                # Get the corresponding bounding box and contour
                label = obj.last_detection.data
                try:
                    bbox = bboxes[label]
                    cntr = contours[label]
                except IndexError:
                    # The object is still considered alive, but no bounding box is associated with it
                    continue

                # Get absolute position and velocity
                abs_pos = obj.estimate.squeeze(axis=0)
                abs_vel = obj.estimate_velocity.squeeze(axis=0)

                # Extract the portion of the frame containing the object
                x1, y1, w, h = bbox
                img = frame[y1:y1+h, x1:x1+w]

                # Declare a new track or update sequences based on estimated information
                if obj.id not in self.tracks:
                    self.tracks[obj.id] = Track(obj.id, abs_pos, abs_vel, cntr, bbox, img)
                else:
                    self.tracks[obj.id].update(abs_pos, abs_vel, cntr, bbox, img)


class Track(object):
    """A container for information related to tracked objects across time."""

    def __init__(self, obj_id, pos, vel, contour, bbox, img):
        """
        Parameters
        ----------
        obj_id: int
            The corresponding object's identifier. This is set by the tracker.

        pos: tuple[int]
            The absolute position of the object at the time of creating the track.

        vel: tuple[int]
            The absolute velocity of the object at the time of creating the track.

        contour: np.array
            The detected contour corresponding to the object at the time of creating the track.

        bbox: np.array
            The detected contour corresponding to the object at the time of creating the track.

        img: np.array
            The portion of the frame containing the object at the time of creating the track.

        """

        # Simply initialize all information sequences
        self.id = obj_id
        self.positions = [pos]
        self.velocities = [vel]
        self.contours = [contour]
        self.bboxes = [bbox]
        self.images = [img]

    def update(self, pos, vel, contour, bbox, img):
        """Stores the given information corresponding to the current time step, in the appropriate sequence.

        Parameters
        ----------
        pos: tuple[int]
            The absolute position of the object at the time of creating the track.

        vel: tuple[int]
            The absolute velocity of the object at the time of creating the track.

        contour: np.array
            The detected contour corresponding to the object at the time of creating the track.

        bbox: np.array
            The detected contour corresponding to the object at the time of creating the track.

        img: np.array
            The portion of the frame containing the object at the time of creating the track.
        """

        # Do what it says on the tin
        self.positions.append(pos)
        self.velocities.append(vel)
        self.contours.append(contour)
        self.bboxes.append(bbox)
        self.images.append(img)


class VideoIterator(cv.VideoCapture):
    """Turns OpenCV's video capture into a fully managed iterator."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __iter__(self):
        return self

    def __next__(self):
        # Make sure the stream is opened
        if not self.isOpened():
            raise StopIteration

        # Read and return the next frame if possible
        ret, frame = self.read()
        if not ret:
            raise StopIteration
        return frame

    def get_metadata(self):
        """Returns information related to the video file or stream itself, such as width, height, or frame rate."""

        width, height = int(self.get(cv.CAP_PROP_FRAME_WIDTH)), int(self.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.get(cv.CAP_PROP_FPS))
        tot_frames = int(self.get(cv.CAP_PROP_FRAME_COUNT))

        return width, height, fps, tot_frames


class Memory(Dataset):
    """Aggregates unique experiences to use for training models."""

    def __init__(self):
        super().__init__()

        # Keep track of inputs added to the dataset to make sure they are unique
        self._in_set = set()
        self._data = None
        self._inpts = None
        self._targs = None

    def add(self, inpt, targ):
        """Adds an (input, target) pair to the dataset, if they are not already part of it."""

        # Computes a hash of the input
        inpt_hash = hashlib.sha256(inpt.tobytes(), usedforsecurity=False)

        # If not a duplicate
        if inpt_hash not in self._in_set:
            # Add the record to the dataset
            inpt = torch.tensor(inpt, dtype=torch.float32).unsqueeze(0)
            targ = torch.tensor(targ, dtype=torch.float32).unsqueeze(0)
            if self._inpts is None:
                self._inpts = inpt
                self._targs = targ
            else:
                self._inpts = torch.vstack([self._inpts, inpt])
                self._targs = torch.vstack([self._targs, targ])
            self._in_set.add(inpt_hash)

    def __len__(self):
        return len(self._in_set)

    def __getitem__(self, idx):
        # Return the corresponding input and target
        return self._inpts[idx], self._targs[idx]


class SoundMapper(nn.Module):
    """Defines a trainable mapping between xenobot movement features, and sound (more specifically, frequency and amplitude)."""

    def __init__(self, nb_ins: int, hidden_lays: list[int], nb_outs: int = 2, l_rate: float = 1e-3, device: str = 'cuda'):
        """Declares a multi-layer perceptron linking input features to frequency and amplitude.

        Parameters
        ----------
            nb_ins: int
                The number of inputs.

            hidden_lays: list[int]
                A list of sizes for the hidden layers. If empty, the input and output layers will be connected directly.

            nb_outs: int, optional
                The number of output layers.

            l_rate: float, optional
                The step size for updating the network's parameters.

            device: {'cuda', 'cpu', 'auto'}, optional
                The type of device to use for computation.

        """

        # Initialize the parent class
        super().__init__()

        # Define the device to use for computation
        if device in ['cuda', 'cpu']:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Define the network's architecture
        if len(hidden_lays) == 0:
            self._linear = nn.Sequential(
                nn.Linear(nb_ins, out_features=nb_outs, device=self._device),
                nn.Sigmoid()
            )
        else:
            self._linear = nn.Sequential()
            in_size = nb_ins
            for hid_size in hidden_lays:
                # Add layer
                self._linear.append(nn.Linear(in_size, hid_size, device=self._device))
                #self._linear.append(nn.ReLU(inplace=True))  # Could be replaced with Tanh()
                self._linear.append(nn.Tanh())
                self._linear.append(nn.LayerNorm(hid_size, device=self._device))
                # Record size for next input
                in_size = hid_size

            # Add output layer
            self._linear.append(nn.Linear(in_size, nb_outs, device=self._device))
            #self._linear.append(nn.Sigmoid())
            self._linear.append(nn.Hardsigmoid())

        # Define optimizer
        self.optim = torch.optim.SGD(self.parameters(), lr=l_rate, momentum=0.9, weight_decay=1e-5, maximize=False)  # Assumes we want to minimize the loss

    def forward(self, x):
        # Make sure the input is on the right device
        if x.device != self._device:
            x = x.to(self._device)

        # Propagate the input through the multi-layer perceptron
        out = self._linear(x).squeeze()

        # Scale and return the frequencies for X and Y coordinates
        return out

    def train_nn(self, weight_path, dset, loss_fn, nb_epoch, batch_size, patience: int = 5):
        """Uses the given dataset and loss function to train the model.
        """

        # Get a dataloader from the provided dataset
        dl = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)

        # Switch the model to training mode
        self = self.train()

        # Train the network
        min_loss = np.inf
        cnt = 0
        for epoch in range(nb_epoch):
            train_loss = 0
            for inpts, targs in dl:
                # Reset the gradient
                self.optim.zero_grad()

                # Extract the inputs and targets
                inpts = inpts.to(self._device)
                targs = targs.to(self._device)

                # Compute the predictions
                preds = self(inpts)

                # Compute and back-propagate loss
                loss = loss_fn(preds, targs)
                loss.backward()
                self.optim.step()

                # Aggregate the loss
                train_loss += loss.item()

            train_loss /= len(dl)

            # Save the weights if the loss decreased
            if train_loss < min_loss:
                min_loss = train_loss
                torch.save(self.state_dict(), weight_path)
                cnt = 0
            else:
                cnt += 1
                if cnt > patience:
                    # Stop training if the loss has been decreasing for some time
                    break

        # Return the loss
        logger.info(f'Min Loss {min_loss}')
        return min_loss

    @property
    def device(self):
        return self._device


class RecurrentSoundMapper(SoundMapper):
    """Defines a trainable recurrent mapping between xenobot movement features and sound."""

    def __init__(self, nb_ins: int, hidden_lays: list[int], nb_lstm: int, size_lstm: int,
                 nb_outs: int = 2, l_rate: float = 1e-3, device: str = 'cuda'):
        """Declares a lstm-based network linking input features to frequency and amplitude.

        Parameters
        ----------
            nb_ins: int
                The number of inputs.

            hidden_lays: list[int]
                A list of sizes for the hidden layers. If empty, the input and output layers will be connected directly.

            nb_lstm: int
                The number of lstm layers to include in the network.

            size_lstm: int
                The size of all lstm layers.

            nb_outs: int, optional
                The number of output layers.

            l_rate: float, optional
                The step size for updating the network's parameters.

            device: {'cuda', 'cpu', 'auto'}, optional
                The type of device to use for computation.

        """

        # Make sure the number and size of LSTM is non-zero
        assert nb_lstm != 0 and size_lstm != 0, 'The number and size of lstm layers cannot be zero. Use a standard `SoundMapper()` if this is what you want.'

        # Initialize the SoundMapper
        super().__init__(size_lstm, hidden_lays, nb_outs, l_rate, device)

        # Declare the recurrent portion of the network
        self._lstm = nn.LSTM(nb_ins, size_lstm, num_layers=nb_lstm, batch_first=True, device=self._device)

        # Redefine the optimizer
        self.optim = torch.optim.SGD(self.parameters(), lr=l_rate, momentum=0.9, weight_decay=1e-5, maximize=False)  # Assumes we want to minimize the loss


    def forward(self, x):
        # Make sure the input is on the same device as the model
        if x.device != self._device:
            x = x.to(self._device)

        # Propagate the input through the lstm
        # Discard the hidden and cell states
        out, _ = self._lstm(x)

        # Return the result of propagating the last hidden state through the linear portion
        return super().forward(out[:, -1, :])


class AttentionSoundMapper(SoundMapper):
    """Defines a trainable attention-based mapping between xenobot movement features and sound."""

    def __init__(self, nb_ins: int, hidden_lays: list[int], nb_heads: int, embed_size: int, nb_outs: int = 2,
                 l_rate: float = 1e-3, device: str = 'cuda'):
        """Declares an attention-based mapper linking input features to frequency and amplitude.

        Parameters
        ----------
            nb_ins: int
                The number of inputs.

            hidden_lays: list[int]
                A list of sizes for the hidden layers. If empty, the input and output layers will be connected directly.

            nb_heads: int
                The number of attention heads to use. Keep in mind that each attention head will have a dimension of `embed_size // nb_heads`.

            embed_size: int
                The size of the embedding vector.
            nb_outs: int, optional
                The number of output layers.

            l_rate: float, optional
                The step size for updating the network's parameters.

            device: {'cuda', 'cpu', 'auto'}, optional
                The type of device to use for computation.

        """

        # Initialize the parent SoundMapper
        super().__init__(embed_size, hidden_lays, nb_outs, l_rate, device)

        # Define embedding and attention layer
        # Assumes an MLP embedding layer
        self._embed = nn.Sequential(
            nn.Linear(nb_ins, embed_size, device=self._device),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_size, device=self._device)
        )

        self._attn = nn.MultiheadAttention(embed_size, nb_heads, batch_first=True, device=self._device)

        # Update the optimizer's definition
        self.optim = torch.optim.SGD(self.parameters(), lr=l_rate, momentum=0.9, weight_decay=1e-5, maximize=False)  # Assumes we want to minimize the loss

    def forward(self, x):
        # Make sure the input is on the same device as the model
        if x.device != self._device:
            x = x.to(self._device)

        # Propagate the input through the embedding and attention layers
        out = self._attn(self.embed(x))

        # Return the frequency and amplitude
        return super().forward(out)
