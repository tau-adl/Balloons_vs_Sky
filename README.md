# Real Time Balloons detection in Videos Using Deep Learning

Deep learning was able to produce significant breakthrough in the field of computer vision in recent years. This, along with the advances in hardware for deep learning allows for running real time object detection from video on hardware mounted on a small drone.

In this project I created an environment setup for training deep neural networks and utilized it to train a neural network to detect balloons from a video. In addition, the balloons detection was optimized to run at real time on the Nvidia Jetson TX2.

## The following is required and helpful for project setup

* The videos for training are available under /datasets/balloons_vs_sky_basic/train
* The video for testing is available under /datasets/balloons_vs_sky_basic/testing
* Project should be run from the projects main directory
* Config file is located under config directory
* All the source files are located under src directory
* Any script can be run with the -h flag to display help messages

## Images generation from videos for training

To generate images from videos for training the following command should be executed:

```
python crop_balloons_from_video.py --input <path to video file>

Exmaple:
python crop_balloons_from_video.py --input datasets/balloons_vs_sky_basic/train/20190320_162139.mp4
```

* A crop is selected with the mouse left button.
* The following keypads control the cropping:

** 'o' - Optimize the cropping region

** 's' - Choose another sky ROI

** 'd' - Start fast cropping

** 'r' - Reset the cropping region

** 'q' - Quit sky ROI

* When the script is in automatic mode, balloons and sky images will be generated automatically

* When the script is in manual mode:

		'5' - Continue to next frame

		'8' - Move rectangle up

		'2' - Move rectangle down

		'4' - Move rectangle left

		'6' - Move rectangle right

		's' - Choose another sky ROI

		'q' - Quit sky ROI

## HDF5 files creation

To build the HDF5 files the following command should be executed:

```
python build_balloons_vs_sky.py
```

To train a CNN on the dataset:
python train_balloons_vs_sky.py --net <CNN architecture> --optimizer <SGD/Adam>

Example:
python train_balloons_vs_sky.py --net NanoVggNet --optimizer SGD --epochs 10 --learning_rate 1e-2






# Real Time Balloons detection in Videos Using Deep Learning

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

