# Usage

#!python3 train_balloons_vs_sky.py --net MiniGoogLeNet --optimizer SGD --epochs 10 --learning_rate 1e-2
#!python3 train_balloons_vs_sky.py --net MiniVggNet    --optimizer SGD --epochs 10 --learning_rate 1e-2
#!python3 train_balloons_vs_sky.py --net NanoVggNet    --optimizer SGD --epochs 10 --learning_rate 1e-2
#!python3 train_balloons_vs_sky.py --net ResNetV1      --optimizer SGD --epochs 10 --learning_rate 1e-1
#!python3 train_balloons_vs_sky.py --net ResNetV2      --optimizer SGD --epochs 10 --learning_rate 1e-1

# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# Import the necessary packages
from config import balloons_vs_sky_config as config
from src.preprocessing import ImageToArrayPreprocessor
from src.preprocessing import SimplePreprocessor
from src.preprocessing import MeanPreprocessor
from src.callbacks import EpochCheckpoint
from src.callbacks import TrainingMonitor
from src.io import HDF5DatasetGenerator
from src.cnn import NanoVGGNet
from src.cnn import MiniVGGNet
from src.cnn import MiniGoogLeNet
from src.cnn import DeepGoogLeNet
from src.cnn import DeeperGoogLeNet
from src.cnn import ResNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import plot_model
import keras.backend as K
import argparse
import json
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--net", type=str, default="DeeperGoogLeNet",
	help="net type to train")
ap.add_argument("-o", "--optimizer", type=str, default="SGD",
	help="optimizer to use")
ap.add_argument("-e", "--epochs", type=int, default=10,
	help="number of epochs to train")
ap.add_argument("-l", "--learning_rate", type=float, default=1e-3,
	help="learning rate for training")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
	help="epoch to restart training at")
ap.add_argument("-d", "--debug", type=int, default=0,
	help="add debug files and prints")
args = vars(ap.parse_args())

# Construct the training image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	# width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	# horizontal_flip=True, fill_mode="nearest")
aug = ImageDataGenerator(rotation_range=30, zoom_range=0.1,
	width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
	horizontal_flip=True, vertical_flip=True, fill_mode="nearest")

	
# Load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# Initialize the image preprocessors
sp = SimplePreprocessor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# Initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=aug,
	preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
	preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

# If there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["start_epoch"] == 0:
	print("[INFO] compiling model...")
	if   args["net"] == "NanoVggNet":
		model = NanoVGGNet.build(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,
			depth=3, classes=config.NUM_CLASSES)
	elif args["net"] == "MiniVggNet":
		model = MiniVGGNet.build(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,
			depth=3, classes=config.NUM_CLASSES)
	elif args["net"] == "MiniGoogLeNet":
		model = MiniGoogLeNet.build(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,
			depth=3, classes=config.NUM_CLASSES)
	elif args["net"] == "DeepGoogLeNet":
		model = DeepGoogLeNet.build(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,
			depth=3, classes=config.NUM_CLASSES, reg=0.0002)
	elif args["net"] == "DeeperGoogLeNet":
		model = DeeperGoogLeNet.build(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT,
			depth=3, classes=config.NUM_CLASSES, reg=0.0002)
	elif args["net"] == "ResNetV1":
		model = ResNet.build(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3, config.NUM_CLASSES,
			(6, 6, 6), (64, 64, 128, 256), reg=0.0005, image_size=config.IMAGE_HEIGHT)
	elif args["net"] == "ResNetV2":
		model = ResNet.build(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3, config.NUM_CLASSES,
			(3, 3, 3), (32, 32, 64, 128), reg=0.0005, image_size=config.IMAGE_HEIGHT)
	elif args["net"] == "ResNet":
		model = ResNet.build(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3, config.NUM_CLASSES,
			(3, 4, 6), (64, 128, 256, 512), reg=0.0005, image_size=config.IMAGE_HEIGHT)
	else:
		raise ValueError("illegal net")
	
	# Write the network architecture visualization graph to disk
	plotPath = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(args["net"])])
	plot_model(model, to_file=plotPath, show_shapes=True)	

	if args["optimizer"] == "SGD":
		opt = SGD(args["learning_rate"], momentum=0.9)
	elif args["optimizer"] == "Adam":
		opt = Adam(args["learning_rate"])
	else:
		raise ValueError("illegal optimizer")
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])

# Otherwise, load the checkpoint from disk
else:
	print("[INFO] loading {}_{}_epoch_{}...".format(args["net"], config.DATASET_NAME, args["start_epoch"]))
	modelPath = os.path.sep.join([config.OUTPUT_PATH, "checkpoints",
		"{}_{}_epoch_{}.hdf5".format(args["net"], config.DATASET_NAME, args["start_epoch"])])
	model = load_model(modelPath)

	# Update the learning rate
	print("[INFO] old learning rate: {}".format(
		K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, args["learning_rate"])
	print("[INFO] new learning rate: {}".format(
		K.get_value(model.optimizer.lr)))

# Construct the callbacks
cpPath = os.path.sep.join([config.OUTPUT_PATH, "checkpoints"])
cpPrefix = "{}_{}".format(args["net"], config.DATASET_NAME)
figPath = os.path.sep.join([config.OUTPUT_PATH,"{}_{}.png".format(args["net"], config.DATASET_NAME)])
jsonPath = os.path.sep.join([config.OUTPUT_PATH, "{}_{}.json".format(args["net"], config.DATASET_NAME)])	
callbacks = [
	EpochCheckpoint(cpPath, prefix=cpPrefix, every=5,
		startAt=args["start_epoch"]),
	TrainingMonitor(figPath, jsonPath=jsonPath,
		startAt=args["start_epoch"])]

model.fit_generator(
	trainGen.generator(),
	validation_data=valGen.generator(),
	steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
	validation_steps=valGen.numImages // config.BATCH_SIZE,
	epochs=args["epochs"],
	max_queue_size=10,
	callbacks=callbacks, verbose=1)

# Close the databases
trainGen.close()
valGen.close()