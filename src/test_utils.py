import contextlib
import time 
import torch

class Detector(object):
    """Define the object that made detection based on the past clips seen."""

    FIRE = 1 # the class of a video that contains fire/smoke
    NO_FIRE = 0 # the class of a video that does not contain fire or smoke



    def __init__(self, clip_len, clip_stride, thresholds, consecutiveness):
        """Initialize the detector.

        Args:
            clip_len (int): length of the clip in frames
            clip_stride (int): stride of the clip in frames
            thresholds (dict): map label -> threshold
            consecutiveness (dict): map label -> number of consecutive clips to classify as label
        """
        # These lists will contain tuples made by (clip_index, confidence)
        self._labels = list(thresholds.keys()) # List of labels in order of fire strongness
        self._classification = self.NO_FIRE # The ACTUALLY classification of the video, default 0
        self._incriminated_frame = None # None if no fire is detected
                                        # A frame index if fire is detected

        # Define a dict that maps label -> list of tuples (clip_index, confidence)
        self._anomaly_clips = {label: [] for label in self._labels}
        
        self.CLIP_LEN = clip_len
        self.CLIP_STRIDE = clip_stride

        self._thresholds = thresholds # Map label -> threshold
        self._consecutiveness = consecutiveness # Map label -> number of consecutive clips to classify as label
        self._consecutiveness_counter = {label: 0 for label in self._labels} # Map label -> counter of consecutive clips classified as label


    def step(self, clip_result, clip_index):
        """Update the state of the detector based on the result of the last clip.

        Args:
            clip_result (torch.tensor): tensor of shape (1, 2) containing the probability of fire and smoke labels
        """
        for i in range(len(self._labels)):
            label = self._labels[i]
            if clip_result[i] >= self._thresholds[label]:
                self._consecutiveness_counter[label] += 1
                self.__add_clip(label, clip_index, clip_result[i])
            else:
                self._consecutiveness_counter[label] = 0
        
        self.__state_update()

    def get_labels(self):
        """Get a list of labels for the current classification.
        The format is a list of strings, each string is a label.

        Raise:
            ValueError: if no fire is detected

        Returns:
            list: list of labels. Example: ["Fire", "Smoke"]
        """
        if self.get_classification() == 0:
            raise ValueError("No anomaly detected")
        video_labels = []
        for label in self._labels:
            if self._consecutiveness_counter[label] >= self._consecutiveness[label]:
                video_labels.append(label)

        return video_labels
        
    def get_classification(self):
        return self._classification
    
    def get_frame(self):
        """Get the incriminated frame."""
        return self._incriminated_frame

    def __add_clip(self, label, clip_index, confidence):
        self._anomaly_clips[label].append((clip_index, confidence))

    def __state_update(self):
        """Update the state of the detector based on the clips seen so far."""
        # A video is classified as fire if 3 clips have labels smoke
        # Or if just a clip has fire label
        
        # if some list of clips is longer of the consecutiveness threshold mark as fire
        for label in self._labels:
            if self._consecutiveness_counter[label] >= self._consecutiveness[label]:
                self._classification = self.FIRE

                # POLICY: Incriminated frame is the last seen frame.
                # the last "consecutiveness threshold" clips

                # Get the last clip index of the last "consecutiveness threshold" clips
                last_clip_index = self._anomaly_clips[label][-1][0]

                # Get the last frame index of the last clip
                self._incriminated_frame = last_clip_index * self.CLIP_STRIDE + self.CLIP_LEN - 1

                return

                
            

class Profile(contextlib.ContextDecorator):
    """
    Profile class for profiling execution time. 
    Can be used as a decorator with @Profile() or as a context manager with 'with Profile():'.

    Attributes
    ----------
    t : float
        Accumulated time.
    cuda : bool
        Indicates whether CUDA is available.

    Methods
    -------
    __enter__()
        Starts timing.
    __exit__(type, value, traceback)
        Stops timing and updates accumulated time.
    time()
        Returns the current time, synchronizing with CUDA if available.
    """
    
    def __init__(self, t=0.0):
        """
        Initializes the Profile class.

        Parameters:
        t : float
            Initial accumulated time. Defaults to 0.0.
        """
        self.t = t  # Accumulated time
        self.cuda = torch.cuda.is_available()  # Checks if CUDA is available

    def __enter__(self):
        """
        Starts timing.
        
        Returns:
        self
        """
        self.start = self.time()  # Start time
        return self

    def __exit__(self, type, value, traceback):
        """
        Stops timing and updates accumulated time.
        
        Parameters:
        type, value, traceback : 
            Standard parameters for an exit method in a context manager.
        """
        self.dt = self.time() - self.start  # Delta-time
        self.t += self.dt  # Accumulates delta-time

    def time(self):
        """
        Returns the current time, synchronizing with CUDA if available.
        
        Returns:
        float
            The current time.
        """
        if self.cuda:  # If CUDA is available
            torch.cuda.synchronize()  # Synchronizes with CUDA
        return time.time()  # Returns current time
