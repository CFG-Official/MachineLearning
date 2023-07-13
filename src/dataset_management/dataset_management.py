import gdown
import os
import shutil
from pathlib import Path

# The default extension for video files. Currently set to ".mp4".
default_video_extension = ".mp4"

# The default extension for annotation files. Currently set to ".rtf".
default_annotation_extension = ".rtf"

# The expected name of the parent directory for videos that do not contain fire.
# If a video file's parent directory has this name, it is categorized as "no fire".
default_no_fire_video_folder_name = "0"

# The expected name of the parent directory for annotations that correspond to videos without fire.
# If an annotation file's parent directory has this name, it corresponds to a "no fire" video.
default_no_fire_annotation_folder_name = "GT_TRAINING_SET_CL0"

# The expected name of the parent directory for videos that contain fire.
# If a video file's parent directory has this name, it is categorized as "fire".
default_fire_video_folder_name = "1"

# The expected name of the parent directory for annotations that correspond to videos with fire.
# If an annotation file's parent directory has this name, it corresponds to a "fire" video.
default_fire_annotation_folder_name = "GT_TRAINING_SET_CL1"


class DatasetEntry:
    """ Represents an entry in the dataset, namely a video and its associated annotation.

    Attributes:
        _video_path (Path): The path to the video.
        _annotation_path (Path): The path to the annotation.
        _name (str): The name of the video, which is also the name of the annotation (excluding the extension).
        _fire (bool): True if the video contains fire, False otherwise.

    Class Variables:
        default_annotation_extension (str): The default extension for the annotation file, currently set to ".rtf".
    """
    __slots__ = ['_video_path', '_annotation_path', '_name', '_fire']

    def __init__(self, video_path: Path, annotation_path: Path, fire: bool = None):
        """ Initializes a DatasetEntry instance.

        Args:
            video_path (Path): The path to the video.
            annotation_path (Path): The path to the annotation.
        """
        DatasetEntry.__check_video_annotation(video_path, annotation_path)
        self._video_path = video_path
        self._annotation_path = annotation_path
        self._name = video_path.stem
        self._fire = fire
        if fire is None:
            self._fire = DatasetEntry.__check_fire(video_path, annotation_path)

    def get_name(self):
        """str: The name of the video and the annotation."""
        return self._name

    def get_video_path(self):
        """Path: The path to the video."""
        return self._video_path

    def get_annotation_path(self):
        """Path: The path to the annotation."""
        return self._annotation_path

    def set_annotation_path(self, annotation_path: Path):
        if self._annotation_path == annotation_path:
            return

        if annotation_path.is_file():
            raise ValueError(f"New path already exist: {annotation_path}")

        if annotation_path.suffix != default_annotation_extension:
            raise ValueError(
                f"New annotation file does not have the default extension ({default_annotation_extension})")

        # Create the directory tree if it does not exist
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self._annotation_path, annotation_path)
        try:
            self.__check_video_annotation(self._video_path, annotation_path)
            os.remove(self._annotation_path)
            self._annotation_path = annotation_path
        except Exception:
            os.remove(annotation_path)
            raise ValueError(f"New annotation file does not correspond to the video: {annotation_path}")

    def set_new_annotation(self, new_annotation_path: Path):
        """
        Sets the annotation path to the new one given as input and replaces the old annotation file with the new one.
        It checks if the video and the new annotation correspond to each other, if not it raises an exception.
        If the new_annotation_path is identical to the existing annotation path, the function simply returns.

        Args:
            new_annotation_path (Path): The path of the new annotation.

        Raises:
            ValueError: If the new annotation file does not correspond to the video.
        """
        # Skip operation if the new annotation path is identical to the current one
        if self._annotation_path == new_annotation_path:
            return

        self.__check_video_annotation(self._video_path, new_annotation_path)
        os.remove(self._annotation_path)
        shutil.copy(new_annotation_path, self._annotation_path)

    def is_fire(self):
        """bool: True if the video contains fire, False otherwise."""
        return self._fire

    def is_mivia(self):
        """
        Returns a boolean indicating if the video is from the MIVIA dataset or not.
        
        According to dataset conventions, videos from the MIVIA dataset that contain fire have a name count less than
        or equal to 261. Similarly, videos without fire from the MIVIA dataset have a name count less than or equal
        to 103. Additionally, videos with "custom" in their name are not from the MIVIA dataset.

        Returns:
            bool: True if the video is from the MIVIA dataset, False otherwise.
        """

        # Extract the video count from the video name

        LAST_FIRE_MIVIA_VIDEO = 261
        LAST_NO_FIRE_MIVIA_VIDEO = 103

        if "custom" in self._video_path.stem.lower() or "video" not in self._video_path.stem.lower():
            return False

        video_count = self._video_path.stem.lower().split("video")[-1]
        video_count = int(video_count.split("_")[0])

        # Check the video count according to the fire condition
        if self.is_fire() and video_count <= LAST_FIRE_MIVIA_VIDEO:
            return True

        if not self.is_fire() and video_count <= LAST_NO_FIRE_MIVIA_VIDEO:
            return True

        return False

    @staticmethod
    def __check_fire(video_path: Path, annotation_path: Path) -> bool:
        """Checks if the video and annotation are categorized as fire.

        This method checks if the parent directory of the video_path is "1" and the parent directory
        of the annotation_path is "GT_TRAINING_SET_CL1", in which case it returns True (indicating fire).
        If the parent directories are "0" and "GT_TRAINING_SET_CL0", respectively, it returns False
        (indicating no fire). In any other case, it raises an exception.

        Args:
            video_path (Path): The path to the video.
            annotation_path (Path): The path to the annotation.

        Returns:
            bool: True if the video and annotation are categorized as fire, False otherwise.

        Raises:
            ValueError: If the parent directories of the video and annotation do not correspond to the
            expected names for fire/no fire categorization.
        """
        video_parent = video_path.parent.name
        annotation_parent = annotation_path.parent.name

        if video_parent == default_no_fire_video_folder_name and annotation_parent == default_no_fire_annotation_folder_name:
            return False
        elif video_parent == default_fire_video_folder_name and annotation_parent == default_fire_annotation_folder_name:
            return True
        else:
            raise ValueError(
                f"Unexpected parent directory names for video ({video_parent}) of video file {video_path} and annotation ({annotation_parent}) of annotation file {annotation_path}")

    @staticmethod
    def __check_video_annotation(video_path: Path, annotation_path: Path):
        """Checks if the video and annotation files exist, have the same names and the right extensions.

        Raises:
            FileNotFoundError: If the video or the annotation file does not exist.
            ValueError: If the video and the annotation file do not have the same name or if the video file is not a .mp4 file.

        Returns:
            bool: False if the annotation file does not have the default extension, True otherwise.
        """
        if not video_path.is_file() or not annotation_path.is_file():
            raise FileNotFoundError("Video or annotation file not found")
        if video_path.suffix != default_video_extension:
            raise ValueError(f"Video file {video_path} is not a {default_video_extension} file!")

        if annotation_path.suffix != default_annotation_extension:
            raise ValueError(f"Annotation file {annotation_path} is not a {default_annotation_extension} file!")

        if video_path.stem != annotation_path.stem:
            raise ValueError(f"Video {video_path} and annotation {annotation_path} have different names")

    def __eq__(self, other):
        """Checks if two DatasetEntry instances are equal.

        Args:
            other (DatasetEntry): The other DatasetEntry instance to compare.

        Returns:
            bool: True if the two instances are equal, False otherwise.
        """
        if not isinstance(other, DatasetEntry):
            return False
        return self._video_path == other._video_path and self._annotation_path == other._annotation_path

    def __lt__(self, other):
        """Compares two DatasetEntry instances based on their name.

        Args:
            other (DatasetEntry): The other DatasetEntry instance to compare.

        Returns:
            bool: True if the current instance's name is less than the other instance's name, False otherwise.
        """
        return self._name < other._name

    def set_name(self, name: str):
        """Changes the name of the video and annotation and their paths, excluding the extension.

        Args:
            name (str): The new name.
        """
        self._name = name

    def move_entry(self, new_video_directory_path: Path, new_annotation_directory_path: Path):
        """Moves the video and annotation to new directory paths and updates their paths.

        Args:
            new_video_directory_path (Path): The new directory path for the video.
            new_annotation_directory_path (Path): The new directory path for the annotation.
        """
        new_video_path = new_video_directory_path / (self._name + self._video_path.suffix)
        new_annotation_path = new_annotation_directory_path / (self._name + self._annotation_path.suffix)
        shutil.move(self._video_path, new_video_path)
        shutil.move(self._annotation_path, new_annotation_path)
        self._video_path = new_video_path
        self._annotation_path = new_annotation_path


class DatasetManagement:
    __slots__ = ['_source_videos_directory', '_source_no_fire_annotations_directory',
                 '_source_fire_annotations_directory', '_destination_directory', '_entries_paths_list']
    _default_output_folder_name = "REORGANIZED_DATASET"

    def __init__(self, source_videos_directory: Path, source_no_fire_annotations_directory: Path,
                 source_fire_annotations_directory: Path, destination_directory: Path):
        self._entries_paths_list = []
        DatasetManagement.__check_directories(source_videos_directory, source_no_fire_annotations_directory,
                                              source_fire_annotations_directory)
        self._source_videos_directory = source_videos_directory
        self._source_no_fire_annotations_directory = source_no_fire_annotations_directory
        self._source_fire_annotations_directory = source_fire_annotations_directory
        destination_directory = destination_directory / self._default_output_folder_name
        print(f"\n====================\nDestination directory: {destination_directory}\n====================\n")

        self._destination_directory = destination_directory
        self._populate_dataset()

    @staticmethod
    def __check_directories(source_videos_directory: Path, source_no_fire_annotations_directory: Path,
                            source_fire_annotations_directory: Path):
        if not source_videos_directory.is_dir():
            raise NotADirectoryError(f"{source_videos_directory} is not a directory")

        if not source_no_fire_annotations_directory.is_dir():
            raise NotADirectoryError(f"{source_no_fire_annotations_directory} is not a directory")

        if not source_fire_annotations_directory.is_dir():
            raise NotADirectoryError(f"{source_fire_annotations_directory} is not a directory")

    def __len__(self):
        return len(self._entries_paths_list)

    def __getitem__(self, index):
        return self._entries_paths_list[index]

    def __setitem__(self, index, value: DatasetEntry):
        self._entries_paths_list[index] = value

    def __delitem__(self, index):
        del self._entries_paths_list[index]

    def __iter__(self):
        return iter(self._entries_paths_list)

    def add_entry(self, video_path: Path, annotation_path: Path):
        new_entry = DatasetEntry(video_path, annotation_path)
        self._entries_paths_list.append(new_entry)

    def remove_entry(self, entry: DatasetEntry):
        self._entries_paths_list.remove(entry)

    def _populate_dataset(self):
        no_fire_videos_dir = self._source_videos_directory / default_no_fire_video_folder_name

        if not no_fire_videos_dir.is_dir():
            raise ValueError(f"Directory not found for no fire videos")

        for video_file in no_fire_videos_dir.glob(f"*{default_video_extension}"):
            corresponding_annotation = self._source_no_fire_annotations_directory / (
                        video_file.stem + default_annotation_extension)
            if corresponding_annotation.is_file():
                self.add_entry(video_file, corresponding_annotation)

        # Check and populate fire entries
        fire_videos_dir = self._source_videos_directory / default_fire_video_folder_name

        if not fire_videos_dir.is_dir() or not self._source_fire_annotations_directory.is_dir():
            raise ValueError(f"Directory not found for fire videos or annotations")

        for video_file in fire_videos_dir.glob(f"*{default_video_extension}"):
            corresponding_annotation = self._source_fire_annotations_directory / (
                        video_file.stem + default_annotation_extension)
            if corresponding_annotation.is_file():
                self.add_entry(video_file, corresponding_annotation)

    def reorganize_dataset(self):
        shutil.rmtree(self._destination_directory, ignore_errors=True)
        os.makedirs(self._destination_directory, exist_ok=True)
        no_fire_video_folder = self._destination_directory / "TRAINING_SET" / default_no_fire_video_folder_name
        fire_video_folder = self._destination_directory / "TRAINING_SET" / default_fire_video_folder_name
        no_fire_annotation_folder = self._destination_directory / default_no_fire_annotation_folder_name
        fire_annotation_folder = self._destination_directory / default_fire_annotation_folder_name
        os.makedirs(no_fire_video_folder, exist_ok=True)
        os.makedirs(fire_video_folder, exist_ok=True)
        os.makedirs(no_fire_annotation_folder, exist_ok=True)
        os.makedirs(fire_annotation_folder, exist_ok=True)

        mivia_fire_count = 0
        mivia_no_fire_count = 0
        custom_fire_count = 0
        custom_no_fire_count = 0

        for entry in self._entries_paths_list:
            if not isinstance(entry, DatasetEntry):
                raise TypeError(f"Entry is not a DatasetEntry instance")
            if entry.is_fire():

                if entry.is_mivia():
                    entry.set_name(f"Video{mivia_fire_count}")
                    mivia_fire_count += 1
                else:
                    entry.set_name(f"custom_Video{custom_fire_count}")
                    custom_fire_count += 1
                entry.move_entry(fire_video_folder, fire_annotation_folder)

            else:

                if entry.is_mivia():
                    entry.set_name(f"Video{mivia_no_fire_count}")
                    mivia_no_fire_count += 1
                else:
                    entry.set_name(f"custom_Video{custom_no_fire_count}")
                    custom_no_fire_count += 1
                entry.move_entry(no_fire_video_folder, no_fire_annotation_folder)

    @staticmethod
    def __get_files_dict(directory: Path, extension: str):
        extension = extension if extension.startswith('.') else f'.{extension}'
        percorsi = {}
        for file in directory.rglob(f'*{extension}'):
            if file.is_file():
                percorsi[file.stem] = file
        return percorsi

    def reload_annotations(self, fire_annotations_directory_new: Path = None, no_fire_annotations_directory_new: Path = None):
        if not fire_annotations_directory_new is None and not fire_annotations_directory_new.is_dir():
            raise ValueError(
                f"Directory of new fire videos:\"{fire_annotations_directory_new}\" not found for fire annotations")

        if not no_fire_annotations_directory_new is None and not no_fire_annotations_directory_new.is_dir():
            raise ValueError(
                f"Directory of new no fire videos:\"{no_fire_annotations_directory_new}\" not found for no fire annotations")

        if not self._source_fire_annotations_directory.is_dir():
            raise ValueError(
                f"Directory of old fire videos:\"{self._source_fire_annotations_directory}\" not found for fire annotations")

        if not self._source_no_fire_annotations_directory.is_dir():
            raise ValueError(
                f"Directory of old no fire videos:\"{self._source_no_fire_annotations_directory}\" not found for no "
                f"fire annotations")

        if not fire_annotations_directory_new is None:
            new_fire_set = self.__get_files_dict(fire_annotations_directory_new, default_annotation_extension)
            new_fire_set_len = len(new_fire_set)
        else:
            new_fire_set_len = 0
        
        if not no_fire_annotations_directory_new is None:
            new_no_fire_set = self.__get_files_dict(no_fire_annotations_directory_new, default_annotation_extension)
            new_no_fire_set_len = len(new_no_fire_set)
        else:
            new_no_fire_set_len = 0
        
        total_count = new_fire_set_len + new_no_fire_set_len

        if total_count != len(self._entries_paths_list):
            print(
                f"WARNING: The number of annotations in the new directories is different from the number of entries in the dataset. The dataset will be reloaded anyway.")

        for entry in self._entries_paths_list:
            if not isinstance(entry, DatasetEntry):
                raise TypeError(f"{entry} is not a DatasetEntry instance")

            if entry.is_fire() and new_fire_set_len > 0:
                entry.set_new_annotation(new_fire_set[entry.get_annotation_path().stem])
            elif not entry.is_fire() and new_no_fire_set_len > 0:
                entry.set_new_annotation(new_no_fire_set[entry.get_annotation_path().stem])

    def count_entries(self):
        no_fire_entries = sum(1 for entry in self._entries_paths_list if not entry.is_fire())
        fire_entries = sum(1 for entry in self._entries_paths_list if entry.is_fire())
        total_entries = len(self._entries_paths_list)

        return no_fire_entries, fire_entries, total_entries


def download_google_file(shader_url, output_name):
    id_url = "https://drive.google.com/uc?id=" + shader_url.split("/")[5]
    gdown.download(id_url, output_name)