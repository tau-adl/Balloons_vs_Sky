# Usage
# python build_balloons_vs_sky.py

# Import packages
from config import balloons_vs_sky_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.preprocessing import AspectAwarePreprocessor
from src.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import json
import cv2
import os
import h5py

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", type=int, default=0,
	help="Add debug files")
args = vars(ap.parse_args())

# Grab the paths to the images, extract the labels and encode them
imagePaths = []
imageLabels = []
paths = list(paths.list_images(config.IMAGES_PATH))
for path in paths:
	label = path.split(os.path.sep)[-2]
	if label != "origin":
		imagePaths.append(path);
		imageLabels.append(label);

if args["debug"]:
	print("[DEBUG] Labels {}".format(imageLabels))

le = LabelEncoder()
imageLabels = le.fit_transform(imageLabels)

if args["debug"]:
	print("[DEBUG] Labels encoded {}".format(imageLabels))
	print("[DEBUG] Dataset size {}".format(len(imageLabels)))

# Perform stratified sampling from the images set to build the
# testing split
split = train_test_split(imagePaths, imageLabels,
	test_size=config.TEST_IMAGES_PCT, stratify=imageLabels,
	random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

# Perform another stratified sampling, this time to build the
# validation data and training data
split = train_test_split(trainPaths, trainLabels,
	test_size=config.VAL_IMAGES_PCT, stratify=trainLabels,
	random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split

# Construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
datasets = [
	("train", trainPaths, trainLabels, config.TRAIN_HDF5),
	("val", valPaths, valLabels, config.VAL_HDF5),
	("test", testPaths, testLabels, config.TEST_HDF5)]

# Initialize the image preprocessor and the lists of RGB channel averages
aap = AspectAwarePreprocessor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
(R, G, B) = ([], [], [])

# Loop over the dataset tuples
for (type, paths, labels, outputPath) in datasets:
	# Create HDF5 writer
	print("[INFO] Building {}...".format(outputPath))
	writer = HDF5DatasetWriter((len(paths), config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3), outputPath)

	# Initialize the progress bar
	widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
		progressbar.Bar(), " ", progressbar.ETA()]
	pbar = progressbar.ProgressBar(maxval=len(paths),
		widgets=widgets).start()

	# Loop over the image paths
	for (idx, (path, label)) in enumerate(zip(paths, labels)):
		# Load the image and process it
		image = cv2.imread(path)
		image = aap.preprocess(image)

		# If we are building the training dataset, then compute the
		# mean of each channel in the image, then update the
		# respective lists
		if type == "train":
			(b, g, r) = cv2.mean(image)[:3]
			R.append(r)
			G.append(g)
			B.append(b)

		# Add the image and label to the HDF5 dataset
		writer.add([image], [label])
		pbar.update(idx)

	# Close the HDF5 writer
	pbar.finish()
	writer.close()

# Construct a dictionary of averages, then serialize the means to a
# JSON file
print("[INFO] Serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()

# Display the shape of the HDF5 files created
filenames = [config.TRAIN_HDF5, config.VAL_HDF5, config.TEST_HDF5]
if args["debug"]:
	print("[DEBUG] HDF5 files {}".format(filenames))
for filename in filenames:
	db = h5py.File(filename, "r")
	print("[INFO] Created HDF5 file {} shape {}".format(filename, db["images"].shape))
	db.close()
