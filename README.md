# Face recognition using Python and OpenCV

## Requirements
Install required libraries:
```
sudo apt-get install build-essential \
    cmake \
    gfortran \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    liblapack-dev \
    libatlas3-base \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy

sudo apt-get clean

sudo apt-get install libopenblas-dev
```

Install the dlib Python library:
```
sudo pip3 install dlib
```

Install the Python face recognition library (https://github.com/ageitgey/face_recognition):
```
sudo pip3 install --no-cache-dir face_recognition
```

Install the imutils Python library:
```
pip install imutils
```

Create a "datasets" folder before running any of the Python files.

## Capturing user images
Open the 01_headshots.py file and change the variable "name" to the name of the user. Create a folder under "datasets" with the same name.
Run the 01_headshots.py file and press space to take a photo. Take about 10 photos of each user.

## Training the face recognition model
Run the 02_train_model.py file. 
This process will output an "encodings.pickle" file with the face encodings for each user. Every time a user is added the model needs to be re-trained.

## Performing face recognition
Run the 03_face_recognition.py file. 
The process will attempt to recognize the face using the camera.
