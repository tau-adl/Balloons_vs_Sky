# Usage

# BATCH_SIZE = 64, IMAGE_SIZE = 32
#!python3 eval_balloons_vs_sky.py --model output/balloons_vs_sky_basic/checkpoints/MiniGoogLeNet_balloons_vs_sky_basic_epoch_10.hdf5 -> Inference took 0.0409 seconds with MiniGoogLeNet
#!python3 eval_balloons_vs_sky.py --model output/balloons_vs_sky_basic/checkpoints/MiniVggNet_balloons_vs_sky_basic_epoch_10.hdf5    -> Inference took 0.0100 seconds with MiniVggNet
#!python3 eval_balloons_vs_sky.py --model output/balloons_vs_sky_basic/checkpoints/NanoVggNet_balloons_vs_sky_basic_epoch_10.hdf5    -> Inference took 0.0055 seconds with NanoVggNet
#!python3 eval_balloons_vs_sky.py --model output/balloons_vs_sky_basic/checkpoints/ResNetV1_balloons_vs_sky_basic_epoch_10.hdf5      -> Inference took 0.0490 seconds with ResNetV1
#!python3 eval_balloons_vs_sky.py --model output/balloons_vs_sky_basic/checkpoints/ResNetV2_balloons_vs_sky_basic_epoch_10.hdf5      -> Inference took 0.0157 seconds with ResNetV2

# BATCH_SIZE = 128, IMAGE_SIZE = 32
#!python3 eval_balloons_vs_sky.py --model output/balloons_vs_sky_basic/checkpoints/MiniGoogLeNet_balloons_vs_sky_basic_epoch_10.hdf5 -> Inference took 0.0782 seconds with MiniGoogLeNet
#!python3 eval_balloons_vs_sky.py --model output/balloons_vs_sky_basic/checkpoints/MiniVggNet_balloons_vs_sky_basic_epoch_10.hdf5    -> Inference took 0.0207 seconds with MiniVggNet
#!python3 eval_balloons_vs_sky.py --model output/balloons_vs_sky_basic/checkpoints/NanoVggNet_balloons_vs_sky_basic_epoch_10.hdf5    -> Inference took 0.0115 seconds with NanoVggNet
#!python3 eval_balloons_vs_sky.py --model output/balloons_vs_sky_basic/checkpoints/ResNetV1_balloons_vs_sky_basic_epoch_10.hdf5      -> Inference took 0.1052 seconds with ResNetV1
#!python3 eval_balloons_vs_sky.py --model output/balloons_vs_sky_basic/checkpoints/ResNetV2_balloons_vs_sky_basic_epoch_10.hdf5      -> Inference took 0.0349 seconds with ResNetV2

# CPU:
# Inference took 1.9969 seconds with DeepGoogLeNet
# Inference took 0.0618 seconds with NanoVggNet

# GPU (Colab):
# Inference took 0.0205 seconds with DeepGoogLeNet
# Inference took 0.0055 seconds with NanoVggNet

# Nvidia Jetson:
# Inference took 0.0395 seconds with NanoVggNet

# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# Import the necessary packages
from config import balloons_vs_sky_config as config
from src.preprocessing import ImageToArrayPreprocessor
from src.preprocessing import SimplePreprocessor
from src.preprocessing import MeanPreprocessor
from src.utils import rank5_accuracy
from src.io import HDF5DatasetGenerator
from keras.models import load_model
import numpy as np
import argparse
import json
import time

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
	help="path to *specific* model checkpoint to load")
ap.add_argument("-d", "--debug", type=int, default=0,
	help="add debug files and prints")
args = vars(ap.parse_args())

# Load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# Initialize the image preprocessors
sp = SimplePreprocessor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# Initialize the testing dataset generators
testGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
	preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

# Load the network
modelName = args["model"]
modelName = modelName.split("/")[-1]
modelName = modelName.split(".")[-2]
modelName = modelName.split("_")[0]
print("[INFO] Loading model {}".format(modelName))
model = load_model(args["model"])

# Make predictions on the testing data
print("[INFO] Predicting on test data...")
predictions = model.predict_generator(
	testGen.generator(),
	steps=testGen.numImages // config.BATCH_SIZE,
	max_queue_size=10)

# Compute rank-1 and rank-5
(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] Rank-1: {:.2f}%".format(rank1 * 100))
# print("[INFO] Rank-5: {:.2f}%".format(rank5 * 100))

# Predict on dummy batch to calculate inference time, first prediction is significantly longer in GPU
dummy_batch  = np.zeros((config.BATCH_SIZE, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3))
_ = model.predict(dummy_batch)
startTime = time.time()
_ = model.predict(dummy_batch)
endTime = time.time()
print("[INFO] Inference took {:.4f} seconds with {}".format(endTime - startTime, modelName))

# Close the database
testGen.close()