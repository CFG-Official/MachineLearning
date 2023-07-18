import cv2, os, argparse, random
import numpy as np
import math


# PERSONAL IMPORTS
from test_utils import Detector
from test_utils import Profile
import torch
import albumentations
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
from models.FireDetectionModelFactory import FireDetectionModelFactory

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
    args = parser.parse_args()
    return args

args = init_parameter()

output_function = nn.Sigmoid()
model_name = "x3d_l"
mode = "single"
pad_strategy = "zeros"
weight_path = Path("../weights/" + str(model_name) + ".pth") # TODO: change this
clip_len = 16
clip_stride = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# If CUDA is avaible, empty the cache to avoid OOM
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if mode == "multi":
    # Define the labels, thresholds and consecutiveness for the detector when it has
    # to work in mode "multi", i.e. when it has to detect both fire and smoke (or none)
    labels = ["Fire", "Smoke"]
    thresholds_map = {"Fire": 0.5, "Smoke": 0.5}
    consecutiveness_map = {"Fire": 1, "Smoke": 3}
elif mode == "single":
    # Define the labels, thresholds and consecutiveness for the detector when it has
    # to work in mode "single", i.e. when it has to detect only fire (or none)
    labels = ["Fire"]
    thresholds_map = {"Fire": 0.5} # TODO: validate it
    consecutiveness_map = {"Fire": 1} # TODO: validate it

FIRE = 1 # the class of a video that contains fire/smoke
NO_FIRE = 0 # the class of a video that does not contain fire or smoke

# Preparing the model
model = FireDetectionModelFactory.create_model(model_name, num_classes=len(labels), to_train=0)
model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
model.eval()
model = model.float().to(device)

# For all the test videos
for video_index, video in enumerate(os.listdir(args.videos)):
    # Process the video
    video_path = os.path.join(args.videos, video)

    ret = True
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video informations
    fps = cap.get(cv2.CAP_PROP_FPS) 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    num_clips = math.ceil(max(1, (total_frames - clip_len)/clip_stride + 1)) # It must generate at least one clip

    clip = []
    frame_counter = 0
    clip_counter = 0
    detector = Detector(clip_len, clip_stride, thresholds_map, consecutiveness_map)

    while ret:
        ret, img = cap.read()
        
        # Here you should add your code for applying your method
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 use a different format for image, BGR, so we need to convert it to RGB 
            # add to the list of frames
            clip.append(np.asarray(img)) # append the frame in list
            frame_counter += 1

            # When a clip is complete, we apply the model to it
            if frame_counter == clip_len:

                input = apply_preprocessing(clip, model.preprocessing)
                input =  torch.stack([transforms.functional.to_tensor(input[k])
                                for k in input.keys()]) 

                input = input.to(device)
                
                with torch.no_grad():
                    out = output_function(model(input))

                # Check if fire is detected basing on past detection
                detector.step(out, clip_counter)
                if detector.get_classification() == FIRE:
                    break

                # The next clip will start from the frame clip_stride
                clip = clip[clip_stride:] # remove the first clip_stride frames

                frame_counter = clip_len - clip_stride
                clip_counter += 1

        ########################################################
    cap.release()

    # Here we are if:
    # 1) the video is finished
    # 2) fire is detected

    # 1)
    # If the video is finished but no fire is detected, check if the last clip is complete
    # If it is not complete, we need to complete it with the last frames
    # After that, a last check for fire detection is performed
    if clip_counter < num_clips and len(clip) > 0 and detector.get_classification() == NO_FIRE: # TODO: check if it is correct

        if pad_strategy == "duplicate":
            # STRATEGY 1: DUPLICATE LAST FRAME
            clip.extend([clip[-1]] * (clip_len - frame_counter))
        elif pad_strategy == "zeros":
            # STRATEGY 2: PAD WITH ZEROS
            print(len(clip))
            clip.extend([np.zeros(clip[0].shape)] * (clip_len - frame_counter)) # PAD FRAME
        else:
            raise ValueError("Invalid padding strategy {}".format(pad_strategy))

        input = apply_preprocessing(clip, model.preprocessing)
        input =  torch.stack([transforms.functional.to_tensor(input[k])
                                for k in input.keys()]) 
        input_tensor = torch.stack(input).to(device)
        with torch.no_grad():
            out = output_function(model(input_tensor))

        # Check if fire is detected basing on past detection
        detector.step(out, clip_counter)
    
    # 2)
    # If fire is detected, write on file the result of the classification
    
    os.makedirs(args.results, exist_ok=True)
    
    video_name, video_ext = os.path.splitext(video)
    results_file = os.path.join(args.results, video_name + ".txt")

    f = open(results_file, "w")
    if detector.get_classification() == FIRE:
        # Print the time of the first frame of the fire
        second_of_detection = math.floor(detector.get_frame()/fps)
        f.write(str(second_of_detection))

    ########################################################
    f.close()