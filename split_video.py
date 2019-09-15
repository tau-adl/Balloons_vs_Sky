# Usage
# python split_video.py --input datasets/balloons_vs_sky/test/VID_20190320_162218.mp4

# Import packages
import numpy as np
import argparse
import cv2
import os

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video file")
ap.add_argument("-d", "--debug", type=int, default=0,
	help="enable debug option")
args = vars(ap.parse_args())

# Parse input video name
videoName = args["input"]
videoName = videoName.split("/")[-1]
videoName = videoName.split(".")[-2]

# Parse output directory name and print
pathToDir = args["input"]
pathToDir = pathToDir.rsplit("/", 1)[0]
outputDir = os.path.sep.join([pathToDir, videoName])
print("[INFO] Output directory is {}".format(outputDir))

# Create a VideoCapture object to read from input video file
cap = cv2.VideoCapture(args["input"])

# Create output folder 
if not os.path.isdir(outputDir):
	os.mkdir(outputDir)

# Init video loop variables
frame_idx = 0

print("[INFO] Processing video")

while (cap.isOpened()):
	# Read frame
	success, frame = cap.read()
	if success == True:
		# Rotate frame 90 degrees clockwise
		frame = np.rot90(frame, 3, (0,1))
		
		# Write frame to output directory
		outputPath = os.path.sep.join([outputDir, "{}.jpg".format(frame_idx)])
		cv2.imwrite(outputPath, frame)

		if args["debug"]:
			print("[DEBUG] Frame {} written".format(frame_idx))

		frame_idx += 1
	else: 
		break

print("[INFO] Done writing {} frames".format(frame_idx))
