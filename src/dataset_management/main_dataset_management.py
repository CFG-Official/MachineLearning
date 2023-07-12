from dataset_management import *
import os
import shutil
import zipfile
import sys
import os
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from project_paths import *



""" 
    Preconditions:
    - videos_link must be a valid link to a zip file containing the following structure:
        - TRAINING_SET (directory)
            - 0 (directory)
                - *.mp4
            - 1 (directory)
                - *.mp4
    
    - annotations_link must be a valid link to a zip file containing the following structure:
        - GT_TRAINING_SET_CL0 (directory)
            - *.rtf (corrisponding to the relative video in TRAINING_SET/0)
        - GT_TRAINING_SET_CL1 (directory)
            - *.rtf (corrisponding to the relative video in TRAINING_SET/1)
    
    
    If you have a local copy of the dataset, you can skip the download simply by placing the videos and annotations in
    two zip files named "TRAINING_SET.zip" and "GT.zip" and placing them in the same directory as this script.
    
    After the execution of the script, the relabeled dataset will be placed in REORGANIZED_DATASET (directory).

"""

videos_link = "https://drive.google.com/file/d/1eTDG_SbHkCo0OeVwRKugQ2vDV2csDx6q/view?usp=sharing"
annotations_link = "https://drive.google.com/file/d/1UjWkvzzezXNOkncas4Q-kP9X9VU2D0OE/view?usp=sharing"
SELECTED_MODE = "REORGANIZE" # "REORGANIZE" or "RELOAD_ANNOTATION

root_path = Path.cwd()
if root_path.name != "MachineLearning":
    main_dir_found = False
    for parent in root_path.parents:
        print("Parent:", parent)
        if parent.name == "MachineLearning":
            root_path = parent
            main_dir_found = True
            break
    if not main_dir_found:
        raise ValueError("Could not find main directory")

zip_videos_path = root_path / "src" / "dataset_management" / "TRAINING_SET.zip"
zip_annotations_path = root_path / "src" / "dataset_management" / "GT.zip"
dataset_management_root = root_path / "src" / "dataset_management"
source_videos_directory = dataset_management_root / "TRAINING_SET"
source_no_fire_annotations_directory = dataset_management_root / default_no_fire_annotation_folder_name
source_fire_annotations_directory = dataset_management_root / default_fire_annotation_folder_name

if not os.path.exists(zip_videos_path):
    download_google_file(videos_link, str(zip_videos_path))

if not os.path.exists(zip_annotations_path):
    download_google_file(annotations_link, str(zip_annotations_path))

if not source_videos_directory.is_dir():
    shutil.rmtree(source_videos_directory, ignore_errors=True)
    with zipfile.ZipFile(zip_videos_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_management_root)

if not source_no_fire_annotations_directory.is_dir() or not source_fire_annotations_directory.is_dir():
    shutil.rmtree(source_no_fire_annotations_directory, ignore_errors=True)
    shutil.rmtree(source_fire_annotations_directory, ignore_errors=True)
    with zipfile.ZipFile(zip_annotations_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_management_root)

destination_directory = dataset_management_root

dataset_management = DatasetManagement(source_videos_directory, source_no_fire_annotations_directory,
                                        source_fire_annotations_directory, destination_directory)
(no_fire_entries, fire_entries, total_entries) = dataset_management.count_entries()
print(f"\n===================\nNo fire count: {no_fire_entries}\nFire count: {fire_entries}\nTotal count: {total_entries}\n===================\n")

if SELECTED_MODE == "REORGANIZE":
    print("\n===================Start Reorganization===================")
    dataset_management.reorganize_dataset()
    print("\n===================End Reorganization===================")
    

elif SELECTED_MODE == "RELOAD_ANNOTATION":
    no_fire_new_annotation_folder = dataset_management_root / "NEW_ANNOTATIONS" / default_no_fire_annotation_folder_name
    dataset_management.reload_annotations(no_fire_annotations_directory_new = no_fire_new_annotation_folder)