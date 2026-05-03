# Smiley - Smile Detection System

This is a Computer Vision project from Umm Al-Qura University, College of Computing, supervised by Dr. Randah Al-Ahmadi. The goal was to build a system that looks at a photo or a webcam feed and decides whether the person is smiling or not.

We split the work across four Python scripts, each handling a different part of the pipeline — training, evaluation, prediction, and live detection.

---

## Dataset Link:
https://www.kaggle.com/datasets/ghousethanedar/smiledetection

## Project Structure

```
computer_vision/
├── train.py
├── evaluate.py
├── predict.py
├── live_camera.py
├── smile_model.h5
├── train_folder/
│   ├── smiling/
│   └── not_smiling/
└── test_folder/
    ├── smiling/
    └── not_smiling/
```

---

## Requirements

This project requires Python 3, The model was tested using Python 3.10.8 and 3.10.20 for windows and pyhton3 on mac, both of which worked without issues. 
On Windows, using newer versions such as Python 3.11 or 3.12 may cause compatibility problems with TensorFlow. 
On macOS, the project runs using the Python 3 command.

---
### Dependencies

# 1. Install dependencies
pip install -r requirements.txt
```
tensorflow>=2.10.0
numpy
matplotlib
scikit-learn
seaborn
Pillow
opencv-python
mediapipe
```

## Setup

Do this once before running anything.

### Windows

Open PowerShell and run each command separately:

```bash
cd "C:\Users\YourName\Downloads\computer_vision"
```

```bash
py -3.10 -m venv venv
```

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

```bash
& "C:\Users\YourName\Downloads\computer_vision\venv\Scripts\Activate.ps1"
```


### Mac

Open Terminal and run each command separately:

```bash
cd ~/Downloads/computer_vision
```

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## Running the Scripts

Every time you open a new terminal, activate the environment first.

**Windows:**
```bash
& "C:\Users\YourName\Downloads\computer_vision\venv\Scripts\Activate.ps1"
cd "C:\Users\YourName\Downloads\computer_vision"
```

**Mac:**
```bash
source ~/Downloads/computer_vision/venv/bin/activate
cd ~/Downloads/computer_vision
```

---

### train.py

```bash
python train.py
```

Trains the MobileNetV2 model on the 2,800 images in train_folder. Training happens in two phases. Phase 1 keeps the MobileNetV2 base frozen and trains only the layers we added on top. Phase 2 unfreezes the last 30 layers and continues training at a lower learning rate. Three callbacks run throughout: ModelCheckpoint saves the best version of the model, EarlyStopping halts training if validation accuracy stops improving for 5 epochs, and ReduceLROnPlateau cuts the learning rate in half when validation loss stalls for 3 epochs.

The final model is saved as smile_model.h5. Skip this step if that file already exists.

---

### evaluate.py

```bash
python evaluate.py
```

Loads smile_model.h5 and runs it on the 1,200 images in test_folder. Predictions above 0.5 are classified as smiling. The script then computes accuracy, precision, recall, and F1-score using scikit-learn and saves two output files: confusion_matrix.png and metrics_chart.png.

---

### predict.py

Single image:
```bash
python predict.py --image path/to/photo.jpg
```

Folder of images:
```bash
python predict.py --folder path/to/folder
```

Unlike evaluate.py, this script does not need images organized into subfolders. It works on any folder of photos. For folder mode, it displays up to 12 predictions in a grid and saves the result as batch_predictions.png. The classification threshold here is 0.4 instead of 0.5, which makes the model slightly more sensitive to smiles.

---

### live_camera.py

```bash
python live_camera.py
```

Opens the webcam. For each frame, it converts the image to grayscale and uses a Haar Cascade detector to find faces. Each detected face gets cropped, resized to 224x224, and passed through the model. A green box with a "Smiling" label appears for scores at or above 0.4, and a red box with "Not Smiling" appears for scores below that. Press Q to exit.

---

## Results

The model was evaluated on 1,200 test images:

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 77.7%  |
| Precision | 77.5%  |
| Recall    | 78.0%  |
| F1-Score  | 77.7%  |

All four metrics fall within 0.5% of each other, which shows the model treats both classes evenly. The false positive and false negative counts were also very close (136 vs. 132), confirming that the model is not leaning toward either class.

---

## Libraries

| Library      | Version | Purpose                             |
|--------------|---------|-------------------------------------|
| TensorFlow   | 2.10.0  | Building and training the model     |
| NumPy        | Latest  | Numerical operations                |
| OpenCV       | Latest  | Camera access and face detection    |
| Matplotlib   | Latest  | Plotting training curves            |
| scikit-learn | Latest  | Evaluation metrics                  |
| Seaborn      | Latest  | Confusion matrix visualization      |

---

## Team

| University ID | Name               |
|---------------|--------------------|
| 444001325     | ليان محمد الحربي   |
| 44410242      | غايه تركي الحازمي  |
| 444002439     | ريداء براك المعبدي |
