# Usage
# python show_hdf5.py --path datasets/balloons_vs_sky_basic/hdf5

# Import packages
import argparse
import os
import h5py
import cv2

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to HDF5 directory")
ap.add_argument("-d", "--debug", type=int, default=0,
	help="Add debug files")
args = vars(ap.parse_args())

files = os.listdir(args["path"])
if args["debug"]:
	print("[DEBUG] HDF5 files {}".format(files))
fullPathFiles = [os.path.sep.join([args["path"], file]) for file in files]
if args["debug"] > 1:
	print("[DEBUG] HDF5 full path files {}".format(fullPathFiles))

for (file, fullPathFile) in zip(files, fullPathFiles):
	db = h5py.File(fullPathFile, "r")
	print("[INFO] HDF5 file {} shape {}".format(file, db["images"].shape))
	if args["debug"] > 2:
		first_image = db["images"][0]
		path = os.path.sep.join([args["path"], "debug_{}_first_image.jpg".format(file)])
		cv2.imwrite(path, first_image)
		print("[DEBUG] saved image debug_{}_first_image.jpg".format(file))
		

		last_image_idx = db["images"].shape[0] - 1
		last_image = db["images"][last_image_idx]

		path = os.path.sep.join([args["path"], "debug_{}_last_image.jpg".format(file)])
		cv2.imwrite(path, last_image)
		print("[DEBUG] saved image debug_{}_last_image.jpg".format(file))
	db.close()