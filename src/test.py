import cv2, os, argparse, random
import numpy as np
import math


# PERSONAL IMPORTS
import torch
import albumentations
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from pathlib import Path
from pytorchvideo.models import create_res_basic_head
from models.FireDetectionModelFactory import FireDetectionModelFactory
from sklearn.metrics import precision_score, recall_score

class Detector(object):
    """Define the object that made detection based on the past clips seen."""

    def __init__(self, clip_len, clip_stride):
        # These lists will contain tuples made by (clip_index, confidence)
        self._fire_clips = []
        self._smoke_clips = []
        self._classification = 0 # 0 = ACTUALLY classified as no fire,
                                # 1 = classified as fire
        self._incriminated_frame = None # None if no fire is detected
                                       # A frame index if fire is detected

        self.SMOKE_THRESHOLD = 0.3
        self.FIRE_THRESHOLD = 0.3
        self.CONSECUTIVE_FIRE_CLIPS = 1
        self.CONSECUTIVE_SMOKE_CLIPS = 3
        self.CLIP_LEN = clip_len
        self.CLIP_STRIDE = clip_stride


    def step(self, clip_result, clip_index):
        """Update the state of the detector based on the result of the last clip.

        Args:
            clip_result (torch.tensor): tensor of shape (1, 2) containing the probability of fire and smoke labels
        """
        if clip_result[0] >= self.FIRE_THRESHOLD:
            self.__add_fire_clip(clip_index, clip_result[0])
        if clip_result[1] >= self.SMOKE_THRESHOLD:
            self.__add_smoke_clip(clip_index, clip_result[1])
        
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
            raise ValueError("No fire detected")
        labels = []
        if len(self._fire_clips) > 0: labels.append("Fire")
        if len(self._smoke_clips) > 0: labels.append("Smoke")

        return labels
        
    def get_classification(self):
        return self._classification
    
    def get_frame(self):
        """Get the incriminated frame."""
        return self._incriminated_frame

    def __add_fire_clip(self, clip_index, confidence):
        self._fire_clips.append((clip_index, confidence))
    
    def __add_smoke_clip(self, clip_index, confidence):
        self._smoke_clips.append((clip_index, confidence))

    def __state_update(self):
        """Update the state of the detector based on the clips seen so far."""
        # A video is classified as fire if 3 clips have labels smoke
        # Or if just a clip has fire label
        
        if len(self._fire_clips) > 0 or len(self._smoke_clips) >= 3:
            self._classification = 1

        if len(self._fire_clips) >= self.CONSECUTIVE_FIRE_CLIPS:
            ### FIRE DETECTED ###
            # The incriminated frame is the center frame of the clip classified as fire

            clip_index = self._fire_clips[-1][0] # Last (and only) clip classified as fire
            # Get the first frame of the clip
            first_frame_clip = clip_index * self.CLIP_STRIDE
            last_frame_clip = first_frame_clip + self.CLIP_LEN
            # The incriminated frame is the center frame of the clip
            self._incriminated_frame = (first_frame_clip + last_frame_clip) // 2
        elif len(self._smoke_clips) >= self.CONSECUTIVE_SMOKE_CLIPS:
            ### SMOKE DETECTED ###
            # The incriminated frame is the first frame of the first clip classified as smoke

            clip_index = self._smoke_clips[0][0] # First clip classified as smoke
            # Get the first frame of the clip
            first_frame_clip = clip_index * self.CLIP_STRIDE
            # The incriminated frame is the first frame of the clip
            self._incriminated_frame = first_frame_clip

def apply_preprocessing(frames, preprocess):
    # Apply preprocessing to list of frames (copy paste of VideoFrameDataset)
    additional_targets = {f"image{i}": "image" for i in range(0, len(frames))}
    preprocess = albumentations.Compose([preprocess], additional_targets=additional_targets)
    transform_input = {"image": frames[0]}
    for i, image in enumerate(frames[1:]):
        transform_input["image%d" % i] = image
    return preprocess(**transform_input)

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    parser.add_argument("--model", type=str, default='x3d_xs', help="Model name")
    parser.add_argument("--clip_len", type=int, default=4, help="Length of a single clip")
    parser.add_argument("--clip_stride", type=int, default=2, help="Stride between clips")
    parser.add_argument("--ground_truth", type=str, default='GT', help="Ground truth folder")
    args = parser.parse_args()
    return args

args = init_parameter()

torch.cuda.empty_cache() # clear memory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = str(args.model)
weight_path = Path("../weights/" + str(args.model) + ".pth")
output_function = nn.Sigmoid()
pad_strategy = "zeros"
labels = ["Fire", "Smoke"]

model = FireDetectionModelFactory.create_model(model_name, num_classes=len(labels), to_train=0)
model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
model.eval()
model = model.float().to(device)

# DEBUG
memory_per_video_occupancy = torch.zeros(len(os.listdir(args.videos)))

# For all the test videos
for video_index, video in enumerate(os.listdir(args.videos)):
    print("-"*50)
    print("Processing video ", video)
    print("-"*50)

    # Process the video
    video_path = os.path.join(args.videos, video)
    
    ret = True
    # Open video file
    cap = cv2.VideoCapture(video_path)
    # Get frame per second 
    fps = cap.get(cv2.CAP_PROP_FPS) 
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    # Get number of clips
    num_clips = math.ceil(max(1, (total_frames - args.clip_len)/args.clip_stride + 1)) # It must generate at least one clip
    
    print('Frames per second =', fps)
    print('Total frames =', total_frames)
    print("Going to generate ", num_clips, " clips")

    # Create a tensor of num_clips elements to store the results
    results = torch.zeros(num_clips, len(labels)).to(device) # TO DO: probabilmente non serve

    clip = []
    frame_counter = 0
    clip_counter = 0
    detector = Detector(args.clip_len, args.clip_stride)

    while ret:
        ret, img = cap.read()
        # Here you should add your code for applying your method
        if ret:
            # add to the list of frames
            clip.append(np.asarray(img)) # append the frame in list
            frame_counter += 1

            # When a clip is complete, we apply the model to it
            if frame_counter == args.clip_len:
                input = apply_preprocessing(clip, model.preprocessing)
                input =  torch.stack([transforms.functional.to_tensor(input[k])
                                for k in input.keys()]) 
                input = input.to(device)
                
                with torch.no_grad():
                    out = output_function(model(input))
                    results[clip_counter] = out

                # The next clip will start from the frame args.clip_stride
                clip = clip[args.clip_stride:] # remove the first args.clip_stride frames
                frame_counter = args.clip_len - args.clip_stride
                clip_counter += 1

                # Check if fire is detected basing on past detection
                detector.step(results[clip_counter-1], clip_counter-1)
                if detector.get_classification() == 1:
                    # Fire detected
                    break
                

        ########################################################
    cap.release()

    # Here we are if:
    # 1) the video is finished
    # 2) fire is detected

    # 1)
    # If the video is finished but no fire is detected, check if the last clip is complete
    # If it is not complete, we need to complete it with the last frames
    # After that, a last check for fire detection is performed
    if clip_counter < num_clips and detector.get_classification() == 0:
        
        if pad_strategy == "duplicate":
            # STRATEGY 1: DUPLICATE LAST FRAME
            clip.extend([clip[-1]] * (args.clip_len - frame_counter))
        elif pad_strategy == "zeros":
            # STRATEGY 2: PAD WITH ZEROS
            clip.extend([np.zeros(clip[0].shape)] * (args.clip_len - frame_counter)) # PAD FRAME
        else:
            raise ValueError("Invalid padding strategy {}".format(pad_strategy))

        input = apply_preprocessing(clip, model.preprocessing)
        input = [transforms.functional.to_tensor(frame) for frame in input.values()]
        input_tensor = torch.stack(input).to(device)
        results[clip_counter] = output_function(model(input_tensor))

        # Check if fire is detected basing on past detection
        detector.step(results[clip_counter], clip_counter)
    
    # 2)
    # If fire is detected, write on file the result of the classification
    f = open(args.results+video.split(".")[0]+".txt", "w")
    if detector.get_classification() == 1:
        # Print the time of the first frame of the fire
        f.write(str(round(detector.get_frame()/fps)))  # NOT NECESSARY -> + "," + ",".join(detector.get_labels()))

    ### DEBUG: Store the memory usage ###
    memory_per_video_occupancy[video_index] = torch.cuda.memory_allocated() / (1024 ** 2)

    ########################################################
    f.close()

#### DEBUG: Compute metrics ####
# For each video in the test set, we take the result file computed by the previous code
# and we compare it with the ground truth file

Y_hat = torch.zeros(len(os.listdir(args.videos))).to(device)
Y = torch.zeros(len(os.listdir(args.videos))).to(device)
delays = []

video_counter = 0
guard_time = 5 # seconds
for video in os.listdir(args.videos):

    # Read the result file
    result_file = open(args.results+video.split(".")[0]+".txt", "r")
    result = result_file.read()
    result_file.close()

    # Read the ground truth file
    gt_file = open(args.ground_truth+video.split(".")[0]+".rtf", "r")
    gt = gt_file.read()
    gt_file.close()

    if len(gt): # if the ground truth file is not empty
        # Fire is present in the video
        # Get the frame in which the fire is present
        g_frame = int(gt.split(",")[0])
        Y[video_counter] = 1 # Fire is present in the video

    if len(result): # if the result file is not empty
        # Fire detected
        # Get the frame in which the fire is detected
        p_frame = int(result.split(",")[0])

        if len(gt):
            # If the fire is detected and present in the video
            # Check if the detection is fast enough
            if p_frame >= max(0, g_frame - guard_time):
                # Detection is fast enough
                Y_hat[video_counter] = 1
                delays.append(abs(p_frame - g_frame))
            else:
                # Detection is not fast enough
                Y_hat[video_counter] = 0 # Not necessary, already initialized to 0

    video_counter += 1

Y = Y.cpu()
Y_hat = Y_hat.cpu()

# Compute precision, recall and f1 score
# Count the number of true positives, false positives and false negatives
tp = 0
fp = 0
fn = 0
tn = 0
for i in range(len(Y)):
    if Y[i] == 1 and Y_hat[i] == 1:
        tp += 1
    elif Y[i] == 0 and Y_hat[i] == 1:
        fp += 1
    elif Y[i] == 1 and Y_hat[i] == 0:
        fn += 1
    else:
        tn += 1

precision = tp/(tp+fp)
recall = tp/(tp+fn)
if len(delays):
    D = sum(delays)/len(delays) 
else:
    print("Can't calculate D because no fire detected in the test set")
Dn = max(0, 60-D)/60

# Print results
print("..:: RESULTS ::..")

print("Accuracy: {:.4f}".format((Y == Y_hat).float().mean().item()))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F-score: {:.4f}".format(2 * precision * recall / (1e-10 + precision + recall)))
print("Detection mean error: {:.4f}".format(D))
print("Detection delay: {:.4f}".format(Dn))
    
print("Final score numerator: {:.4f}".format(precision * recall * Dn))

print("Mean memory usage for computation of each video: {:.4f} MB".format(memory_per_video_occupancy.mean()))

