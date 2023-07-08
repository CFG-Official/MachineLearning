import os
import re
import cv2
import math
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.project_paths import *

def extract_text_from_rtf(file_path):
    with open(file_path, 'r') as file:
        rtf_content = file.read()

    if rtf_content.__contains__("\\rtf1"):
        # Trova il testo tra '\f0\fs24 \cf0 ' e '}'
        match = re.search(r'\\f0\\fs24 \\cf0 (.*?)\}', rtf_content)

        if match:
            extracted_text = match.group(1)
            return extracted_text.strip()
        else:
            return None
    else: 
        return rtf_content


video_folder = train_videos_path / "1"
gt_folder = train_original_annotations_path / "1"

video_list = os.listdir(video_folder)

for video in video_list:
    # Get frame rate
    cap = cv2.VideoCapture(os.path.join(video_folder, video))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Get video annotation
    label = extract_text_from_rtf(os.path.join(gt_folder, video.split(".")[0] + ".rtf"))

    # Get frame number
    frame_number = int(label.split(",")[0])

    # Get other info
    other_info = label.split(",")[1:] # es. ['Smoke', 'Fire']

    # Replace the content of the file rtf
    with open(os.path.join(gt_folder, video.split(".")[0] + ".rtf"), 'w') as file:
        file.write(str(math.ceil(frame_number*frame_rate)) + "," + ",".join(other_info))



