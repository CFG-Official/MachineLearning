# Fire Detection Competition

Welcome to the Fire Detection Competition repository! This competition aims to develop and evaluate models for detecting fire in videos. The competition provides a dataset containing video files along with corresponding annotations indicating the presence of fire and/or smoke.

## Competition Overview
The goal of the competition is to develop robust machine learning models capable of accurately detecting fire in videos. Participants are required to train their models using the provided dataset and evaluate their performance using specified metrics.

## Dataset
The dataset provided for the competition consists of video files in the `.mp4` format, along with annotations indicating the presence of fire and/or smoke in each video. The dataset is divided into training, validation, and test sets to facilitate model development and evaluation.

## Competition Components
The competition repository contains several components to facilitate model development, training, testing, and evaluation. Here's an overview of each component:

### 1. Data Preprocessing
- The `utils/load_datasetV2.ipynb` notebook is responsible for downloading the dataset, organizing it into training, validation, and test sets, and extracting frames from videos.
- Dataset versions are stored on Google Drive, and the notebook provides functionality to download specific versions.
- Annotations are relabeled to indicate the absolute frame where fire or smoke is detected.
- Frames are organized into folders by video, and a customizable frame sampling rate is applied.

### 2. Model Development and Training
- The `FireDetectionModelFactory.py` module provides a factory for creating instances of fire detection models.
- Model training is performed using several notebooks:
  - `train.ipynb`: Implements classic training with binary classification.
    - Hyperparameters are loaded from a JSON file.
    - Dataset splitting is performed to create training and validation sets.
    - Models are trained using epochs, and the best model is saved based on validation loss.
  - `train_k_fold.ipynb`: Implements stratified K-fold cross-validation training with binary classification.
    - Utilizes a custom `K_FOLD` class for dataset splitting and evaluation.
  - `train_multi_label.ipynb`: Implements classic training with multi-label classification.
    - Adjustments are made for multi-label classification, including loss function selection and preprocessing.
  - `train_k_fold_multi_label.ipynb`: Implements stratified K-fold cross-validation training with multi-label classification.
    - Extends the `K_FOLD` class to handle multi-label datasets.

### 3. Testing
- The `test.py` script executes the testing phase, where trained models are applied to video data to detect fire.
- Implements frame extraction, model inference, and video-level classification using the `Detector` class.
- Padding strategies are applied to ensure proper classification of incomplete clips.

### 4. YOLO Training (Optional)
- YOLO (You Only Look Once) training is supported, although it was not the final choice for the competition.
- Pre-trained YOLO models of various sizes can be used for fire detection.

## Usage
1. Clone the repository to your local machine.
2. Download the dataset and organize it using the `load_datasetV2.ipynb` notebook.
3. Train your models using the provided notebooks.
4. Test your models using the `test.py` script.
5. Evaluate your model's performance using the `test_metrics.py` script.

## Additional Notes
- Ensure that all dependencies are installed before running the notebooks and scripts.
- Refer to the documentation within each notebook and script for detailed instructions on usage and functionality.

## Group members
| Surname and Name      | E-Mail                                                                    | Unique ID   |
|-----------------------|---------------------------------------------------------------------------|-------------|
| Cerasuolo Cristian    | [c.cerasuolo2@studenti.unisa.it](mailto:c.cerasuolo2@studenti.unisa.it)   | 0622701899  |
| Ferrara Grazia        | [g.ferrara75@studenti.unisa.it](mailto:g.ferrara75@studenti.unisa.it)     | 0622701901  |
| Guarini Alessio       | [a.guarini7@studenti.unisa.it](mailto:a.guarini7@studenti.unisa.it)       | 0622702042  |
