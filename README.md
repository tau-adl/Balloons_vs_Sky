# Real Time Balloons detection in Videos Using Deep Learning

Deep learning was able to produce significant breakthrough in the field of computer vision in recent years. This, along with the advances in hardware for deep learning allows for running real time object detection from video on hardware mounted on a small drone.

In this project I created an environment setup for training deep neural networks and utilized it to train a neural network to detect balloons from a video. In addition, the balloons detection was optimized to run at real time on the Nvidia Jetson TX2.

## General description of project structure

* The videos for training are available under /datasets/balloons_vs_sky_basic/train
* The video for testing is available under /datasets/balloons_vs_sky_basic/testing
* Project should be run from the projects main directory
* Config file is located under config directory
* All the source files are located under src directory
* Any script can be run with the -h flag to display help messages

## Images generation from videos for training

To generate images from videos for training the following command should be executed:

*python crop_balloons_from_video.py --input \\<path to video file\\>*

Example:

```
python crop_balloons_from_video.py --input datasets/balloons_vs_sky_basic/train/20190320_162139.mp4
```

Elaboration on how to use the cropping tool:

* A crop is selected with the mouse left button.

* The following keypads control the cropping:

  * 'o' - Optimize the cropping region

  * 's' - Choose another sky ROI

  * 'd' - Start fast cropping

  * 'r' - Reset the cropping region

  * 'q' - Quit sky ROI

* When the script is in automatic mode, balloons and sky images will be generated automatically

* When the script is in manual mode:

  * '5' - Continue to next frame

  * '8' - Move rectangle up

  * '2' - Move rectangle down

  * '4' - Move rectangle left

  * '6' - Move rectangle right

  * 's' - Choose another sky ROI

  * 'q' - Quit sky ROI

## HDF5 files creation

To build the HDF5 files the following command should be executed:

*python build_balloons_vs_sky.py*

## Training

To train a CNN on the dataset the following command should be executed:

*python train_balloons_vs_sky.py --net \<CNN architecture\> --optimizer \<SGD/Adam\>*

Example:

```
python train_balloons_vs_sky.py --net NanoVggNet --optimizer SGD --epochs 10 --learning_rate 1e-2
```

## Evaluation

To evaluate a trained CNN on the dataset the following command should be executed:

*python eval_balloons_vs_sky.py – model \<path to HDF5 checkpoint file\>*

Example:

```
python eval_balloons_vs_sky.py --model output/balloons_vs_sky_basic/checkpoints/NanoVggNet_balloons_vs_sky_basic_epoch_10.hdf5 
```

## Detection

To run the balloons detection the following command should be executed:

*python detect_balloons.py --input \<path to input video file or folder of images\> --model \<path to HDF5 checkpoint file\>*

Example for running the detection on an input video:

```
python detect_balloons.py --input datasets/balloons_vs_sky_basic/test/VID_20190320_162218.mp4 --model output/balloons_vs_sky_basic/checkpoints/NanoVggNet_balloons_vs_sky_basic_epoch_10.hdf5
```

Example for running the detection on an input folder of images:

```
python detect_balloons.py --input datasets/balloons_vs_sky_basic/test/VID_20190320_162218 --model output/balloons_vs_sky_basic/checkpoints/NanoVggNet_balloons_vs_sky_basic_epoch_10.hdf5
```

## Additional scripts

To split a video into a folder of images the following command should be executed:

*python split_video.py --input \<path to input video file\>*

Example:

```
python split_video.py --input datasets/balloons_vs_sky/test/VID_20190320_162218.mp4
```

To display the content of the HDF5 files the following command should be executed:

*python show_hdf5.py --path \<path to a folder which contains HDF5 files\>*

Example:

```
python show_hdf5.py --path datasets/balloons_vs_sky_basic/hdf5
```

## Versions

Python 3.6.3

Keras 2.2.4


