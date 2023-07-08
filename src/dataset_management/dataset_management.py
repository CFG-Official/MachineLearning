from pathlib import Path
import shutil, os, zipfile, gdown

# The default extension for video files. Currently set to ".mp4".
_default_video_extension = ".mp4"

# The default extension for annotation files. Currently set to ".rtf".
_default_annotation_extension = ".rtf"

# The expected name of the parent directory for videos that do not contain fire.
# If a video file's parent directory has this name, it is categorized as "no fire".
_default_no_fire_video_folder_name = "0"

# The expected name of the parent directory for annotations that correspond to videos without fire.
# If an annotation file's parent directory has this name, it corresponds to a "no fire" video.
_default_no_fire_annotation_folder_name = "GT_TRAINING_SET_CL0"

# The expected name of the parent directory for videos that contain fire.
# If a video file's parent directory has this name, it is categorized as "fire".
_default_fire_video_folder_name = "1"

# The expected name of the parent directory for annotations that correspond to videos with fire.
# If an annotation file's parent directory has this name, it corresponds to a "fire" video.
_default_fire_annotation_folder_name = "GT_TRAINING_SET_CL1"




class DatasetEntry:
    """ Represents an entry in the dataset, namely a video and its associated annotation.

    Attributes:
        _video_path (Path): The path to the video.
        _annotation_path (Path): The path to the annotation.
        _name (str): The name of the video, which is also the name of the annotation (excluding the extension).
        _fire (bool): True if the video contains fire, False otherwise.

    Class Variables:
        _default_annotation_extension (str): The default extension for the annotation file, currently set to ".rtf".
    """
    __slots__ = ['_video_path', '_annotation_path', '_name', '_fire']

    def __init__(self, video_path: Path, annotation_path: Path, fire: bool = None):
        """ Initializes a DatasetEntry instance.

        Args:
            video_path (Path): The path to the video.
            annotation_path (Path): The path to the annotation.
        """
        DatasetEntry.__check_init(video_path, annotation_path)
        self._video_path = video_path
        self._annotation_path = annotation_path
        self._name = video_path.stem
        self._fire = fire
        if fire is None:
            self._fire = DatasetEntry.__check_fire(video_path, annotation_path)

    def name(self):
        """str: The name of the video and the annotation."""
        return self._name

    def video_path(self):
        """Path: The path to the video."""
        return self._video_path

    def annotation_path(self):
        """Path: The path to the annotation."""
        return self._annotation_path
    
    def is_fire(self):
        """bool: True if the video contains fire, False otherwise."""
        return self._fire
    
    def is_mivia(self):
        """bool: True if the video is from the MIVIA dataset, False otherwise."""
        video_count = self._video_path.stem.split("Video")[1]
        video_count = int(video_count.split("_")[0])
        if "custom" in self._video_path.stem:
            return False
        
        if self.is_fire():
            if video_count > 261:
                return False
            else:
                return True
        
        elif not self.is_fire():
            if video_count > 103:
                return False
            else:
                return True
        
        raise ValueError(f"Cannot classify video: {self._video_path}")



    
    
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

        if video_parent == _default_no_fire_video_folder_name and annotation_parent == _default_no_fire_annotation_folder_name:
            return False
        elif video_parent == _default_fire_video_folder_name and annotation_parent == _default_fire_annotation_folder_name:
            return True
        else:
            raise ValueError(f"Unexpected parent directory names for video ({video_parent}) and annotation ({annotation_parent})")

    
    
    @staticmethod
    def __check_init(video_path: Path, annotation_path: Path):
        """Checks if the video and annotation files exist, have the same names and the right extensions.

        Raises:
            FileNotFoundError: If the video or the annotation file does not exist.
            ValueError: If the video and the annotation file do not have the same name or if the video file is not a .mp4 file.

        Returns:
            bool: False if the annotation file does not have the default extension, True otherwise.
        """
        if not video_path.is_file() or not annotation_path.is_file():
            raise FileNotFoundError("Video or annotation file not found")
        if video_path.stem != annotation_path.stem:
            raise ValueError(f"Video {video_path} and annotation file have different names")

        if video_path.suffix != ".mp4":
            raise ValueError(f"Video file {video_path} is not a {_default_video_extension} file!")

        if annotation_path.suffix != _default_annotation_extension:
            raise ValueError(f"Annotation file {annotation_path} is not a {_default_annotation_extension} file!")
        

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
    __slots__ = ['_source_videos_directory', '_source_no_fire_annotations_directory', '_source_fire_annotations_directory' ,'_destination_directory', '_entries_paths_list']
    _default_output_folder_name = "REORGANIZED_DATASET"

    def __init__(self, source_videos_directory: Path, source_no_fire_annotations_directory: Path, source_fire_annotations_directory: Path ,destination_directory: Path):
        self._entries_paths_list = []
        DatasetManagement.__check_directories(source_videos_directory, source_no_fire_annotations_directory, source_fire_annotations_directory)
        self._source_videos_directory = source_videos_directory
        self._source_no_fire_annotations_directory = source_no_fire_annotations_directory
        self._source_fire_annotations_directory = source_fire_annotations_directory
        destination_directory = destination_directory / self._default_output_folder_name
        print(f"\n====================\nDestination directory: {destination_directory}\n====================\n")

        self._destination_directory = destination_directory
        self._populate_dataset()
        
    @staticmethod
    def __check_directories(source_videos_directory: Path, source_no_fire_annotations_directory: Path, source_fire_annotations_directory: Path):
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
        """Populates the dataset with the entries in the entries_paths_list attribute."""
        
        # Check and populate no fire entries
        no_fire_videos_dir = self._source_videos_directory / _default_no_fire_video_folder_name
        
        if not no_fire_videos_dir.is_dir():
            raise ValueError(f"Directory not found for no fire videos")
        
        for video_file in no_fire_videos_dir.glob(f"*{_default_video_extension}"):
            corresponding_annotation = self._source_no_fire_annotations_directory / (video_file.stem + _default_annotation_extension)
            if corresponding_annotation.is_file():
                self.add_entry(video_file, corresponding_annotation)
        
        # Check and populate fire entries
        fire_videos_dir = self._source_videos_directory / _default_fire_video_folder_name
        
        if not fire_videos_dir.is_dir() or not self._source_fire_annotations_directory.is_dir():
            raise ValueError(f"Directory not found for fire videos or annotations")
        
        for video_file in fire_videos_dir.glob(f"*{_default_video_extension}"):
            corresponding_annotation = self._source_fire_annotations_directory / (video_file.stem + _default_annotation_extension)
            if corresponding_annotation.is_file():
                self.add_entry(video_file, corresponding_annotation)
    
    
    def reorganize_dataset(self):
        """Reorganizes the dataset by moving the videos and annotations to the destination directory."""
        shutil.rmtree(self._destination_directory, ignore_errors=True)
        os.makedirs(self._destination_directory, exist_ok=True)
        no_fire_video_folder = self._destination_directory / "TRAINING_SET" / _default_no_fire_video_folder_name
        fire_video_folder = self._destination_directory / "TRAINING_SET" / _default_fire_video_folder_name
        no_fire_annotation_folder = self._destination_directory / _default_no_fire_annotation_folder_name
        fire_annotation_folder = self._destination_directory / _default_fire_annotation_folder_name
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
            
            


    def count_entries(self):
        """Counts the number of entries for fire and no fire videos and returns these counts along with the total entries.

        Returns:
            no_fire_entries (int): Number of entries with no fire videos.
            fire_entries (int): Number of entries with fire videos.
            total_entries (int): Total number of entries.
        """
        no_fire_entries = sum(1 for entry in self._entries_paths_list if not entry.is_fire())
        fire_entries = sum(1 for entry in self._entries_paths_list if entry.is_fire())
        total_entries = len(self._entries_paths_list)

        return no_fire_entries, fire_entries, total_entries



def download_google_file(shader_url, output_name):
    id_url = "https://drive.google.com/uc?id=" + shader_url.split("/")[5]
    gdown.download(id_url, output_name)

if __name__ == "__main__":

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
    
    
    root_path = Path.cwd()
    if root_path.name != "FireDetection":
        main_dir_found = False
        for parent in root_path.parents:
            print("Parent:", parent)
            if parent.name == "FireDetection":
                root_path = parent
                main_dir_found = True
                break
        if not main_dir_found:
            raise ValueError("Could not find main directory")
            
    zip_videos_path = root_path / "src" / "dataset_management" / "TRAINING_SET.zip"
    zip_annotations_path = root_path / "src" / "dataset_management" / "GT.zip"
    dataset_management_root = root_path / "src" / "dataset_management"
    source_videos_directory = dataset_management_root / "TRAINING_SET"
    source_no_fire_annotations_directory = dataset_management_root / _default_no_fire_annotation_folder_name
    source_fire_annotations_directory = dataset_management_root / _default_fire_annotation_folder_name

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

    dataset_management = DatasetManagement(source_videos_directory, source_no_fire_annotations_directory, source_fire_annotations_directory, destination_directory)
    (no_fire_entries,fire_entries,total_entries) = dataset_management.count_entries()
    print("\n===================\nBefore reorganization\n===================")
    dataset_management.reorganize_dataset()
    print(f"\n===================\nNo fire count: {no_fire_entries}\nFire count: {fire_entries}\nTotal count: {total_entries}\n===================\n")
