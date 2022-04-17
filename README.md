# CIS365-ObjectDetection

Jake Irons & Connor Boerma

### Requirements

- You must be running Python 3.7 or greater.
- Upgrade to the latest version of pip (we used 22.0.4)
- It is **highly** suggested you test this model with GPU support. We had troubles running the model on computers without it. You need the following for GPU support:
  - An NVIDIA GPU (GTX 650 or newer)
  - CUDA Toolkit 11.2. [Windows Installation Instructions can be found here.](https://docs.nvidia.com/cuda/archive/11.2.2/cuda-installation-guide-microsoft-windows/index.html) [Linux installation instructions can be found here.](https://docs.nvidia.com/cuda/archive/11.2.2/cuda-installation-guide-linux/index.html)
  - CuDNN 8.1.0. [Documentation on how to install CuDNN can be found here.](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
- Protocol Buffer Compiler. You can see instructions on how to install [at this link.](https://grpc.io/docs/protoc-installation/)

### How to Run

1. **Make sure to clone this repository with the following command**: `git clone --recurse-submodules -j8 https://github.com/ironsj/CIS365-ObjectDetection.git`.
2. It is suggested that you create and activate your prefered virtual environment for Python. Navigate to the project directory and do so.
3. From within the _TensorFlow/models/research_ directory enter the following command `protoc object_detection/protos/*.proto --python_out=.`
4. Then run the following commands

```
    cp object_detection/packages/tf2/setup.py .
    python -m pip install .
```

5. Change back to the root directory of the project. Then run `pip install -r requirements.txt`.

### How To Pass Images into the Model
1. From within the root directory, locate the *model_scripts* directory. Within that directory, locate *picture.py*.
2. For any images you would like to pass into the model, insert the path to them in the *IMAGE_PATH* array near the top of the file. We already have some images found in the *input* directory which can be found in the root directory. You may use your own images if you wish. Then, save the file after editing.
3. From the root directory, run `python model_scripts/picture.py`.

### How To Pass Videos into the Model
1. From within the root directory, locate the *model_scripts* directory. Within that directory, locate *picture.py*.
2. For any video you would like to pass into the model, set *PATH_TO_VIDEO* equal to a string containing the path to your video. You will need to import your own video into the project, as we could not push the large video files to Github. Then, save the file after editing. **WE ONLY TRIED RUNNING THIS SCRIPT WITH .MP4 FILES. WE SUGGEST YOU DO THE SAME AS WE DO NOT KNOW WHAT THE RESULTS WILL BE LIKE OTHERWISE**.
3. From the root directory, run `python model_scripts/video.py`.
To use your webcam with the model, just run `python model_scripts/webcam.py`.

We received our training and testing data from the [Udacity Self Driving Car Dataset](https://public.roboflow.com/object-detection/self-driving-car).
