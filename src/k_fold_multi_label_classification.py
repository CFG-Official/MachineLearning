import os, shutil, random
from project_paths import *
from k_fold import K_FOLD

class K_FOLD_multi_label(K_FOLD):
    
    def __init__(self, train_splitted_frames_path, train_splitted_annotations_path, val_splitted_frames_path, val_splitted_annotations_path, num_folds = 10):
        # this function comes from the extended class
        super().__init__(train_splitted_frames_path, train_splitted_annotations_path, val_splitted_frames_path, val_splitted_annotations_path, num_folds)

    def create_splits(self):
        # this function modifies the one fo the extended class in order to adapt to the multi-label case
        # now we will split in 4 lists the folders depending if the videos are nothing (00), fire (10), smoke (01) or both (11)
        # to see to which class a video belongs we will check the content of the rtf file in the annotations folder, if it is empty
        # the video is nothing, if it contains only the word "fire" it is fire, if it contains only the word "smoke" it is smoke, 
        # if it contains both it is both
        self.folders_00 = []
        self.folders_10 = []
        self.folders_01 = []
        self.folders_11 = []

        for folder in os.listdir(self.train_splitted_annotations_path / "0"):
            video_folder_name = folder.replace(".rtf", ".mp4")
            self.folders_00.append(video_folder_name)

        for folder in os.listdir(self.train_splitted_annotations_path / "1"):
            video_folder_name = folder.replace(".rtf", ".mp4")
            with open(self.train_splitted_annotations_path / "1" / folder) as f:
                content = f.read()
                if "Fire" in content and "Smoke" in content:
                    self.folders_11.append(video_folder_name)
                elif "Fire" in content:
                    self.folders_10.append(video_folder_name)
                else:
                    self.folders_01.append(video_folder_name)

        random.shuffle(self.folders_00)
        random.shuffle(self.folders_10)
        random.shuffle(self.folders_01)
        random.shuffle(self.folders_11)

        self.folders_00 = [self.folders_00[i::self.num_folds] for i in range(self.num_folds)]
        self.folders_10 = [self.folders_10[i::self.num_folds] for i in range(self.num_folds)]
        self.folders_01 = [self.folders_01[i::self.num_folds] for i in range(self.num_folds)]
        self.folders_11 = [self.folders_11[i::self.num_folds] for i in range(self.num_folds)]

    def split(self):

        os.makedirs(os.path.join(self.val_splitted_frames_path,"0"), exist_ok=True)
        os.makedirs(os.path.join(self.val_splitted_frames_path,"1"), exist_ok=True)
        os.makedirs(os.path.join(self.val_splitted_annotations_path,"0"), exist_ok=True)
        os.makedirs(os.path.join(self.val_splitted_annotations_path,"1"), exist_ok=True)
        
        if self.executed_splits == 0:
            for folder in self.folders_00[self.executed_splits]:
                shutil.move(self.train_splitted_frames_path / "0" / folder, self.val_splitted_frames_path / "0" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.train_splitted_annotations_path / "0" / annotation, self.val_splitted_annotations_path / "0" / annotation)
            for folder in self.folders_10[self.executed_splits]:
                shutil.move(self.train_splitted_frames_path / "1" / folder, self.val_splitted_frames_path / "1" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.train_splitted_annotations_path / "1" / annotation, self.val_splitted_annotations_path / "1" / annotation)
            for folder in self.folders_01[self.executed_splits]:
                shutil.move(self.train_splitted_frames_path / "1" / folder, self.val_splitted_frames_path / "1" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.train_splitted_annotations_path / "1" / annotation, self.val_splitted_annotations_path / "1" / annotation)
            for folder in self.folders_11[self.executed_splits]:
                shutil.move(self.train_splitted_frames_path / "1" / folder, self.val_splitted_frames_path / "1" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.train_splitted_annotations_path / "1" / annotation, self.val_splitted_annotations_path / "1" / annotation)
        else:
            for folder in self.folders_00[self.executed_splits-1]:
                shutil.move(self.val_splitted_frames_path / "0" / folder, self.train_splitted_frames_path / "0" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.val_splitted_annotations_path / "0" / annotation, self.train_splitted_annotations_path / "0" / annotation)
            for folder in self.folders_10[self.executed_splits-1]:
                shutil.move(self.val_splitted_frames_path / "1" / folder, self.train_splitted_frames_path / "1" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.val_splitted_annotations_path / "1" / annotation, self.train_splitted_annotations_path / "1" / annotation)
            for folder in self.folders_01[self.executed_splits-1]:
                shutil.move(self.val_splitted_frames_path / "1" / folder, self.train_splitted_frames_path / "1" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.val_splitted_annotations_path / "1" / annotation, self.train_splitted_annotations_path / "1" / annotation)
            for folder in self.folders_11[self.executed_splits-1]:
                shutil.move(self.val_splitted_frames_path / "1" / folder, self.train_splitted_frames_path / "1" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.val_splitted_annotations_path / "1" / annotation, self.train_splitted_annotations_path / "1" / annotation)
            for folder in self.folders_00[self.executed_splits]:
                shutil.move(self.train_splitted_frames_path / "0" / folder, self.val_splitted_frames_path / "0" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.train_splitted_annotations_path / "0" / annotation, self.val_splitted_annotations_path / "0" / annotation)
            for folder in self.folders_10[self.executed_splits]:
                shutil.move(self.train_splitted_frames_path / "1" / folder, self.val_splitted_frames_path / "1" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.train_splitted_annotations_path / "1" / annotation, self.val_splitted_annotations_path / "1" / annotation)
            for folder in self.folders_01[self.executed_splits]:
                shutil.move(self.train_splitted_frames_path / "1" / folder, self.val_splitted_frames_path / "1" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.train_splitted_annotations_path / "1" / annotation, self.val_splitted_annotations_path / "1" / annotation)
            for folder in self.folders_11[self.executed_splits]:
                shutil.move(self.train_splitted_frames_path / "1" / folder, self.val_splitted_frames_path / "1" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.train_splitted_annotations_path / "1" / annotation, self.val_splitted_annotations_path / "1" / annotation)

        self.executed_splits += 1

if __name__ == "__main__":
    # this is just a test to see if the code works
    k_fold = K_FOLD_multi_label(train_splitted_frames_path = train_splitted_frames_path, train_splitted_annotations_path = train_splitted_annotations_path, val_splitted_frames_path = val_splitted_frames_path, val_splitted_annotations_path = val_splitted_annotations_path, num_folds = 10)
    k_fold.reset()
    # k_fold.create_splits()
    # # print the splits
    # print("folders_00: " + str(k_fold.folders_00))
    # print("folders_10: " + str(k_fold.folders_10))
    # print("folders_01: " + str(k_fold.folders_01))
    # print("folders_11: " + str(k_fold.folders_11))
    # for i in range(10):
    #     k_fold.split()
    #     input("Press Enter to continue...")
    #     print("split number " + str(i+1) + " executed")
