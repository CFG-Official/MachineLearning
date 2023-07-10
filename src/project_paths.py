from pathlib import Path

def find_fire_detection_dir():
    path = Path.cwd()
    if path.name == "MachineLearning":
        return path
    for parent in path.parents:
        if parent.name == "MachineLearning":
            return parent
    raise Exception("MachineLearning directory not found")


root_path = find_fire_detection_dir()

project_root = find_fire_detection_dir()
data_folder_path = project_root / "data"
test_data_folder_path = project_root / "test_data"

## LEVEL 1
videos_path = data_folder_path / "VIDEOS"
original_frames_path = data_folder_path / "ORIGINAL_FRAMES"
original_annotations_path = data_folder_path / "ORIGINAL_GT"
splitted_frames_path = data_folder_path / "SPLITTED_FRAMES"
splitted_annotations_path = data_folder_path / "SPLITTED_GT"

test_data_videos_path = test_data_folder_path / "TEST_VIDEOS"

#### LEVEL 2
train_splitted_frames_path = splitted_frames_path / "TRAINING_SET"
val_splitted_frames_path = splitted_frames_path / "VALIDATION_SET"
test_splitted_frames_path = splitted_frames_path / "TEST_SET"

train_splitted_annotations_path = splitted_annotations_path / "TRAINING_SET"
val_splitted_annotations_path = splitted_annotations_path / "VALIDATION_SET"
test_splitted_annotations_path = splitted_annotations_path / "TEST_SET"

train_videos_path = videos_path / "TRAINING_SET"

train_original_frames_path = original_frames_path / "TRAINING_SET"
train_original_annotations_path = original_annotations_path / "TRAINING_SET"