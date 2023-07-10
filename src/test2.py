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

def classify_from_results(results, stride = 2, clip_len = 3, consecutive_clips = 3, threshold = 0.5, fire_percentage = 0.01):
    """Classification based on clip combined results.
    If there are 10% of clips classified as fire, and 3 of them consecutively, then the video is classified as fire.
    The indicted frame is the first common frame between the first and second clip classified as fire.
    A clip is classified as fire if the probability of fire is greater than 0.75."""
    fire_clips = torch.where(results >= threshold)[0].size()[0]

    # Calculate the percentage of clips classified as fire
    classified_fire_percentage = fire_clips / results.size()[0]
    print("Fire clips: ", classified_fire_percentage)

    if classified_fire_percentage < fire_percentage:
        print("Not enough fire clips")
        return 0, None

    # Enough fire clips, check if there are 3 consecutive clips
    # If there are, return the first frame of the first clip
    for i in range(results.size()[0] - consecutive_clips + 1):
        if torch.all(results[i:i+consecutive_clips] > threshold):
            return 1, i*stride # return the first frame of the first clip

    print("Not enough consecutive fire clips")    
    return 0, None

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    parser.add_argument("--model", type=str, default='slowfast', help="Model name")
    parser.add_argument("--clip_len", type=int, default=3, help="Length of a single clip")
    parser.add_argument("--clip_stride", type=int, default=2, help="Stride between clips")
    args = parser.parse_args()
    return args

args = init_parameter()

# Here you should initialize your method
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = str(args.model)
path = Path("../weights/" + str(args.model) + ".pth")
output_function = nn.Sigmoid()

model = FireDetectionModelFactory.create_model(model_name, num_classes=1, to_train=0)
model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
model.eval()
model = model.float().to(device)



# For all the test videos
for video in os.listdir(args.videos):
    print("Processing video ", video)

    # Process the video
    video_path = os.path.join(args.videos, video)
    ret = True
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_clips = math.ceil(max(1, (total_frames - args.clip_len)/args.clip_stride + 1)) # It must generate at least one clip
    
    print('Frames per second =', fps)
    print('Total frames =', total_frames)
    print("Going to generate ", num_clips, " clips")

    # Create a tensor of num_clips elements to store the results
    results = torch.zeros(num_clips, 1).to(device)

    clip = []
    frame_counter = 0
    clip_counter = 0

    while ret:
        ret, img = cap.read()
        # Here you should add your code for applying your method
        if ret:
            # add to the list of frames
            clip.append(np.asarray(img)) # append the frame in list
            frame_counter += 1
            if frame_counter == args.clip_len:
                input = apply_preprocessing(clip, model.preprocessing)
                # input = [transforms.functional.to_tensor(frame) for frame in input.values()]
                input =  torch.stack([transforms.functional.to_tensor(input[k])
                                for k in input.keys()]) 
                input = input.to(device)
                
                with torch.no_grad():
                    results[clip_counter] = output_function(model(input))

                clip = clip[args.clip_stride:] # remove the first args.clip_stride frames
                frame_counter = args.clip_len - args.clip_stride
                clip_counter += 1

        ########################################################
    cap.release()

    if clip_counter < num_clips:
        # The last clip is not complete, so we need to complete it with the last frames
        # We add the last frame args.clip_len - frame_counter times

        """clip.extend([clip[-1]] * (args.clip_len - frame_counter)) # DUPLICATE FRAME"""
        clip.extend([np.zeros(clip[0].shape)] * (args.clip_len - frame_counter)) # PAD FRAME

        input = apply_preprocessing(clip, model.preprocessing)
        input = [transforms.functional.to_tensor(frame) for frame in input.values()]
        input_tensor = torch.stack(input).to(device)
        results[clip_counter] = output_function(model(input_tensor))

    f = open(args.results+video+".txt", "w")
    
    # Combine results for each clip using a certain criterion
    classification = classify_from_results(results, stride=args.clip_stride, clip_len=args.clip_len, 
                                            consecutive_clips=1, threshold=0.5, fire_percentage=0)
    if classification[1] is None:
        # cast classification[0] to string
        f.write(str(classification[0]))
    else:
        # TO DO: stampare frame indicato
        f.write(str(classification[0]) + " " + str(classification[1]))

    
    ## DEBUG ##
    f.write("\n")
    start_sec = 0
    end_sec = args.clip_len/fps
    for i in range(results.size()[0]):
        string = "Clip {} [{}:{}]: {}".format(i, round(start_sec,2), round(end_sec,2), round(results[i].item(),4))
        start_sec += args.clip_stride/fps
        end_sec = start_sec + args.clip_len/fps
        f.write(string + "\n")



    ########################################################
    f.close()
