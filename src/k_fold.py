import os, shutil
import random
import project_paths

class K_FOLD:
    
    def __init__(self, train_splitted_frames_path, train_splitted_annotations_path, val_splitted_frames_path, val_splitted_annotations_path, num_folds = 10):
        self.train_splitted_frames_path, self.val_splitted_frames_path = train_splitted_frames_path, val_splitted_frames_path
        self.train_splitted_annotations_path, self.val_splitted_annotations_path = train_splitted_annotations_path, val_splitted_annotations_path
        self.num_folds = num_folds
        self.executed_splits = 0
        self.reset()
        self.create_splits()
        
    def create_splits(self):
        """ 
        This function creates the splits for the k-fold cross validation.
        It creates two lists: one for the folders contained in the folder 0 of the training set and another
        for the folders contained in the folder 1 of the training set.
        Each list is splitted into k sublists, where k is the number of folds.
        """
        self.folders_0 = os.listdir(self.train_splitted_frames_path / "0")
        self.folders_1 = os.listdir(self.train_splitted_frames_path / "1")
        random.shuffle(self.folders_0)
        random.shuffle(self.folders_1)
        self.folders_0 = [self.folders_0[i::self.num_folds] for i in range(self.num_folds)]
        self.folders_1 = [self.folders_1[i::self.num_folds] for i in range(self.num_folds)]
    
    def split(self):
        """ 
        This function is called at each iteration of the k-fold cross validation.
        If the validation folder is empty, it simply moves the folders whose name is contained in the list of the current fold
        from the training folder to the validation folder and the corresponding annotations.
        Otherwise it moves the folders from the validation folder to the training folder and the corresponding annotations and then 
        moves the folders from the training folder to the validation folder and the corresponding annotations.
        """
        os.makedirs(os.path.join(self.val_splitted_frames_path,"0"), exist_ok=True)
        os.makedirs(os.path.join(self.val_splitted_frames_path,"1"), exist_ok=True)
        os.makedirs(os.path.join(self.val_splitted_annotations_path,"0"), exist_ok=True)
        os.makedirs(os.path.join(self.val_splitted_annotations_path,"1"), exist_ok=True)
        
        if self.executed_splits == 0:
            for folder in self.folders_0[self.executed_splits]:
                shutil.move(self.train_splitted_frames_path / "0" / folder, self.val_splitted_frames_path / "0" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.train_splitted_annotations_path / "0" / annotation, self.val_splitted_annotations_path / "0" / annotation)
            for folder in self.folders_1[self.executed_splits]:
                shutil.move(self.train_splitted_frames_path / "1" / folder, self.val_splitted_frames_path / "1" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.train_splitted_annotations_path / "1" / annotation, self.val_splitted_annotations_path / "1" / annotation)
        else:
            for folder in self.folders_0[self.executed_splits-1]:
                shutil.move(self.val_splitted_frames_path / "0" / folder, self.train_splitted_frames_path / "0" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.val_splitted_annotations_path / "0" / annotation, self.train_splitted_annotations_path / "0" / annotation)
            for folder in self.folders_1[self.executed_splits-1]:
                shutil.move(self.val_splitted_frames_path / "1" / folder, self.train_splitted_frames_path / "1" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.val_splitted_annotations_path / "1" / annotation, self.train_splitted_annotations_path / "1" / annotation)
            for folder in self.folders_0[self.executed_splits]:
                shutil.move(self.train_splitted_frames_path / "0" / folder, self.val_splitted_frames_path / "0" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.train_splitted_annotations_path / "0" / annotation, self.val_splitted_annotations_path / "0" / annotation)
            for folder in self.folders_1[self.executed_splits]:
                shutil.move(self.train_splitted_frames_path / "1" / folder, self.val_splitted_frames_path / "1" / folder)
                annotation = folder.split(".")[0] + ".rtf"
                shutil.move(self.train_splitted_annotations_path / "1" / annotation, self.val_splitted_annotations_path / "1" / annotation)
                
        self.executed_splits += 1
        
    def reset(self):
        """ 
        Moves all the folders and the annotations from the validation folder to the training folder.
        """
        if os.path.exists(self.val_splitted_frames_path / "0"):
            for folder in os.listdir(self.val_splitted_frames_path / "0"):
                if len(os.listdir(self.val_splitted_frames_path / "0" / folder)) > 0:
                    shutil.move(self.val_splitted_frames_path / "0" / folder, self.train_splitted_frames_path / "0" / folder)
                    annotation = folder.split(".")[0] + ".rtf"
                    if os.path.exists(self.val_splitted_annotations_path / "0") and len(os.listdir(self.val_splitted_annotations_path / "0")) > 0:
                        shutil.move(self.val_splitted_annotations_path / "0" / annotation, self.train_splitted_annotations_path / "0" / annotation)
                
        if os.path.exists(self.val_splitted_frames_path / "1"):
            for folder in os.listdir(self.val_splitted_frames_path / "1"):
                if len(os.listdir(self.val_splitted_frames_path / "1" / folder)) > 0:
                    shutil.move(self.val_splitted_frames_path / "1" / folder, self.train_splitted_frames_path / "1" / folder)
                    annotation = folder.split(".")[0] + ".rtf"
                    if os.path.exists(self.val_splitted_annotations_path / "1") and len(os.listdir(self.val_splitted_annotations_path / "1")) > 0:
                        shutil.move(self.val_splitted_annotations_path / "1" / annotation, self.train_splitted_annotations_path / "1" / annotation)
        
        shutil.rmtree(self.val_splitted_frames_path, ignore_errors=True)
        shutil.rmtree(self.val_splitted_annotations_path, ignore_errors=True)
        
        self.executed_splits = 0
        
if __name__ == "__main__":
    kfold = K_FOLD(project_paths.train_splitted_frames_path, project_paths.train_splitted_annotations_path, project_paths.val_splitted_frames_path, project_paths.val_splitted_annotations_path)
    for i in range(kfold.num_folds):
        kfold.split()
        print("Split number " + str(i+1) + " done")
    
    