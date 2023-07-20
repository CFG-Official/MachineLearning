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
    parser.add_argument("--model", type=str, default='x3d_xs', help="Model name")
    args = parser.parse_args()
    return args

args = init_parameter()

output_function = nn.Sigmoid()
model_name = args.model
mode = "single"
pad_strategy = "duplicate"
weight_path = Path("../weights/" + str(model_name) + ".pth")
clip_len = 16
clip_stride = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.empty_cache() # clear memory

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
    thresholds_map = {"Fire": 0.5}
    consecutiveness_map = {"Fire": 1}

# Preparing the model
model = FireDetectionModelFactory.create_model(model_name, num_classes=len(labels), to_train=0)
model.load_state_dict(torch.load(weight_path, map_location=torch.device("cpu")))
model.eval()
model = model.float().to(device)

## METRICS UTILS ##
processed_frames = 0
computation_time = 0.0
dt = Profile()
memory_per_video_occupancy = torch.zeros(len(os.listdir(args.videos)))
fps_dict = {}

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
    fps_dict[video] = fps
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    # Get number of clips
    num_clips = math.ceil(max(1, (total_frames - clip_len)/clip_stride + 1)) # It must generate at least one clip
    
    print('Frames per second =', fps)
    print('Total frames =', total_frames)
    print("Going to generate ", num_clips, " clips")

    # Create a tensor of num_clips elements to store the results
    results = torch.zeros(num_clips, len(labels)).to(device) # TO DO: probabilmente non serve

    clip = []
    frame_counter = 0
    clip_counter = 0
    detector = Detector(clip_len, clip_stride, thresholds_map, consecutiveness_map)

    while ret:
        ret, img = cap.read()
        
        # Here you should add your code for applying your method
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                    with dt:
                        out = output_function(model(input))
                    computation_time += dt.dt
                    results[clip_counter] = out
                
                # Update frame processed
                processed_frames += clip_len

                # Check if fire is detected basing on past detection
                detector.step(results[clip_counter], clip_counter)
                if detector.get_classification() == 1:
                    break

                # The next clip will start from the frame args.clip_stride
                clip = clip[clip_stride:] # remove the first args.clip_stride frames

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

    # Second condition to avoid bug of empty clip
    if clip_counter < num_clips and len(clip) > 0 and detector.get_classification() == 0:
        print(frame_counter)
        
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
        input = [transforms.functional.to_tensor(frame) for frame in input.values()]
        input_tensor = torch.stack(input).to(device)
        results[clip_counter] = output_function(model(input_tensor))

        # Check if fire is detected basing on past detection
        detector.step(results[clip_counter], clip_counter)
    
    # 2)
    # If fire is detected, write on file the result of the classification
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    f = open(args.results+video.split(".")[0]+".txt", "w")
    if detector.get_classification() == 1:
        # Print the time of the first frame of the fire
        f.write(str(round(detector.get_frame()/fps)))  # NOT NECESSARY -> + "," + ",".join(detector.get_labels()))

    ### METRIC: Store the memory usage ###
    memory_per_video_occupancy[video_index] = torch.cuda.memory_allocated() / (1024 ** 2)

    ########################################################
    f.close()

#### METRIC: Compute metrics ####
# For each video in the test set, we take the result file computed by the previous code
# and we compare it with the ground truth file

tp = 0
fp = 0
fn = 0
tn = 0
delays = []

video_counter = 0
GUARD_TIME = 5 # seconds
MEM_TARGET = 4000 # MB
PFR_TARGET = 10 

results_folder = "../test_data/GT_TEST_SET_SPLIT/"
for video in os.listdir(args.videos):

    # Read the result file
    result_file = open(args.results+video.split(".")[0]+".txt", "r")
    result = result_file.read()
    result_file.close()

    # Read the ground truth file
    gt_file = open(results_folder+video.split(".")[0]+".rtf", "r")
    gt = gt_file.read()
    gt_file.close()

    # TP: all the detections in(P)ositive videos for which 𝑝 ≥ 𝑚𝑎𝑥(0, 𝑔 − 𝛥𝑡)
    # FP: all the detections occurringat any time in (N)egative videos
    #  or in (P)ositivevideos for which 𝑝 < 𝑚𝑎𝑥(0, 𝑔 − 𝛥𝑡)
    # FN: the set of positive videosfor which no fire detection occurs
    if len(gt) and len(result):
        # Fire is present in the video and fire is detected
        g_frame = int(gt.split(",")[0])//fps_dict[video]
        p_frame = int(result.split(",")[0]) # NOT NECESSARY BECAUSE ONLY FRAME IN RESULT FILE
        if p_frame >= max(0, g_frame - GUARD_TIME):
            # Detection is fast enough
            delays.append(abs(p_frame - g_frame))
            tp += 1
        else:
            # Detection is not fast enough
            print("Detection is not fast enough")
            fp += 1
    elif len(result) and not len(gt):
        # Fire is not present in the video and fire is detected
        fp += 1
    elif len(gt) and not len(result):
        # Fire is present in the video and fire is not detected
        fn += 1
    elif not len(gt) and not len(result):
        # Fire is not present in the video and fire is not detected
        tn += 1
    else:
        raise ValueError("Something went wrong")


    video_counter += 1


# Compute precision, recall and f1 score
# Count the number of true positives, false positives and false negatives

try:
    precision = tp/(tp+fp)
except ZeroDivisionError:
    precision = 0
try:
    recall = tp/(tp+fn)
except ZeroDivisionError:
    recall = 0

try:
    D = sum(delays)/len(delays) 
    Dn = max(0, 60-D)/60
except:
    print("Can't calculate D because no fire detected in the test set")
    D = float("inf")
    Dn = 0

f_score = 2 * precision * recall / (1e-10 + precision + recall)
pfr = 1 /(computation_time / processed_frames)
mem = memory_per_video_occupancy.mean().item()

pfr_delta = max(0, PFR_TARGET/pfr - 1)
mem_delta = max(0, mem/MEM_TARGET - 1)
fds = (precision * recall * Dn) / ((1 + pfr_delta) * (1 + mem_delta))
accuracy = (tp + tn) / (tp + tn + fp + fn)

# Print results
print("..:: RESULTS ::..")

print("TP: {}, FP: {}, FN: {}, TN: {}".format(tp, fp, fn, tn))
print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F-score: {:.4f}".format(f_score))
print("Average notification delay: {:.4f}".format(D))
print("Normalized average detection delay: {:.4f}".format(Dn))
print("Processing frame rate: {:.4f}".format(pfr))
print("Memory usage: {:.4f}".format(mem))
print("Final detection score: {:.4f}".format(fds))


import csv
# Write results on csv file
with open(args.results+"metrics.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["accuracy", "precision", "recall", "f-score", "and", "nand", "pfr", "mem","fds"])
    writer.writerow([accuracy, precision, recall, f_score, D, Dn, pfr, mem, fds])