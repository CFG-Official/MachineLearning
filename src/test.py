import cv2, os, argparse, random
import numpy as np
import math


# PERSONAL IMPORTS
import torch
from tqdm import tqdm
import torch.nn as nn
import albumentations
import torchvision.models as models
from torchvision import transforms
from pathlib import Path
from pytorchvideo.models import create_res_basic_head
from models.FireDetectionModelFactory import FireDetectionModelFactory

# Here you should import your method
def apply_preprocessing(frames, preprocess):
    # Apply preprocessing to list of frames (copy paste of VideoFrameDataset)
    additional_targets = {f"image{i}": "image" for i in range(0, len(frames))}
    preprocess = albumentations.Compose([preprocess], additional_targets=additional_targets)
    transform_input = {"image": frames[0]}
    for i, image in enumerate(frames[1:]):
        transform_input["image%d" % i] = image
    return preprocess(**transform_input)

def classify_from_results(results, stride = 2, clip_len = 3, consecutive_clips = 3, threshold = 0.75, fire_percentage = 0.01):
    """Classification based on clip combined results.
    If there are 10% of clips classified as fire, and 3 of them consecutively, then the video is classified as fire.
    The indicted frame is the first common frame between the first and second clip classified as fire.
    A clip is classified as fire if the probability of fire is greater than 0.75."""
    results = np.array(list(results))
    fire_clips = np.where(results > threshold)[0].size

    if fire_clips / results.size < fire_percentage:
        # Not enough fire clips
        print("Not enough fire clips")
        return 0, None
    # Enough fire clips, check if there are 3 consecutive clips
    for i in range(results.size - consecutive_clips + 1):
        if np.all(results[i:i+consecutive_clips] > threshold):
            # 3 consecutive clips
            # return 1 and the first common frame between clip i and i+1
            # clip i starts at frame i*CLIP_STRIDE
            # clip i+1 starts at frame (i+1)*CLIP_STRIDE
            # the first common frame is (i+1)*CLIP_STRIDE
            return 1, (i+1)*stride
    
    print("Not enough consecutive fire clips")


def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    parser.add_argument("--clip_len", type=int, default=3, help="Length of a single clip")
    parser.add_argument("--clip_stride", type=int, default=2, help="Stride between clips")
    args = parser.parse_args()
    return args

args = init_parameter()

# Here you should initialize your method

model_name = "slowfast"
path = Path("../weights/SlowFast L 0.15 A 0.975/best_model.pth")
output_function = nn.Sigmoid()

model = FireDetectionModelFactory.create_model(model_name, num_classes=1, to_train=0)
model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
model.eval()

# For all the test videos
for video in os.listdir(args.videos):
    # Process the video
    video_path = os.path.join(args.videos, video)
    ret = True
    cap = cv2.VideoCapture(video_path)

    print('frames per second =', cap.get(cv2.CAP_PROP_FPS))

    frames = [] # list of frames in video
    while ret:
        ret, img = cap.read()
        # Here you should add your code for applying your method
        if ret:
            # add to the list of frames
            frames.append(np.asarray(img)) # append the frame in list
        ########################################################
    cap.release()

    print("Extracted frames from ", video_path, ": ", len(frames))

    # Apply preprocessing of the model 
    frames = apply_preprocessing(frames, model.preprocessing) 

    # After preprocessing frames is a dict of tensors
    # frames["image"] is the first frame
    # frames["image0"] is the second frame
    # frames["image(num_frames-2)"] is the last frame

    f = open(args.results+video+".txt", "w")

    # Here you should add your code for writing the results
    num_frames = len(frames)
    num_clips = math.ceil(max(1, (num_frames - args.clip_len)/args.clip_stride + 1)) # It must generate at least one clip
    to_pad = (num_clips * args.clip_len) - num_frames # Frames to generate in padding
    
    # Create each clip

    # Create a list of num_clips elements
    clips = [[] for _ in range(num_clips)]

    ordered_items = list(sorted(frames.items(), key=lambda x: 0 if x[0] == "image" else int(x[0][5:])))

    start_frame_counter = 0
    for i in range(num_clips):
        for k in range(args.clip_len):
            try:
                frame_tensor = transforms.functional.to_tensor(ordered_items[start_frame_counter + k][1])
            except IndexError:
                print("Generating padding frame")
                frame_tensor = torch.transpose(transforms.functional.to_tensor(np.zeros(clips[0][0].shape)), 0, 1)
            clips[i].append(frame_tensor)
        start_frame_counter += args.clip_stride


    print("Created ", i+1, "clips out of ", num_clips)

    # Clips[i] è una lista di tensori, ogni tensore è un frame
    # clips[i][j] è il j-esimo frame della i-esima clip
    # Crea un tensore che contiene tutti i frame di una clip

    # Create a dict of results for each clip
    results = {}
    model = model.float()
    # Apply tqdm to this loop

    for i in tqdm(range(num_clips)):
        input = torch.stack(clips[i], dim=0).unsqueeze(0)
        result = output_function(model(input.float()))
        results[i] = result.item()
    
    # Combine results for each clip using a certain criterion
    classification = classify_from_results(results.values(), stride=args.clip_stride, clip_len=args.clip_len, consecutive_clips=3, threshold=0.75, fire_percentage=0.1)
    if classification[1] is None:
        f.write(str(classification[0]))
    else:
        # TO DO: stampare frame indicato
        f.write(str(classification[0]) + " " + str(classification[1]))

    
    ## DEBUG ##
    f.write("\n")
    f.write(str(results.items()))


    ########################################################
    f.close()
