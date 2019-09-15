# Usage
# python detect_balloons.py --input datasets/balloons_vs_sky_basic/test/VID_20190320_162218.mp4 --model output/balloons_vs_sky_basic/checkpoints/NanoVggNet_balloons_vs_sky_basic_epoch_10.hdf5

# Single frame:
# CPU 35.3 seconds - With debug
# GPU 6.3  seconds - With debug
# CPU 24.1 seconds - Without debug
# GPU 3.1  seconds - Without debug

# Two frames:
# CPU 0.061 fps - Last batch is not 64
# GPU 5.127 fps - Last batch is not 64
# CPU 0.091 fps - Last batch is 64
# GPU 4.833 fps - Last batch is 64
# CPU 0.054 fps - Fix fps calculation issue
# GPU 1.945 fps - Fix fps calculation issue

# Optimization:
# Increased downscale
# Added second pass
# Added early termination in first pass
# CPU 0.134 fps - Two frames
# GPU 8.040 fps - Two frames
# GPU 5.820 fps - Entire video

# Optimization:
# Used NanoVGGNet
# Removed second pass
# Applied correct preprocessing
# Updated first pass parameters
# ROI optimization
# CPU 5.512  fps - 100 frames
# GPU 26.958 fps - 100 frames

# Optimization:
# Added tracking
# Added track_info class
# CPU 150 fps - 100 frames
# GPU 170 fps - 100 frames
# CPU 170 fps - All frames
# GPU 190 fps - All frames

# Optimization:
# Updated tracking parameters
# Updated ROI optimization
# CPU 190 fps - All frames
# GPU 220 fps - All frames

# Fix reading from folder
# CPU 170 fps - All frames
# GPU 180 fps - All frames

# Final results:
# # Read video:
# # # CPU           - 194 fps
# # # GPU (Colab)   - 194 fps
# # # Nvidia Jetson - 73 fps
# # Read images from folder:
# # # CPU           - 221 fps
# # # GPU (Colab)   - 186 fps
# # # Nvidia Jetson - 73 fps

# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# Import packages
from config import balloons_vs_sky_config as config
from src.preprocessing import ImageToArrayPreprocessor
from src.preprocessing import SimplePreprocessor
from src.preprocessing import MeanPreprocessor
from src.utils import image_pyramid
from src.utils import sliding_window
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import json
import time
import cv2
import os

NUM_OF_FRAMES = 0 # Set to zero to run on entire video

FORMAT = "mp4" # avi
FOURCC = "mp4v" # "XVID"

BALLOONS_LABLE = 0

INTERPOLATION = cv2.INTER_CUBIC
# INTERPOLATION = cv2.INTER_LINEAR

# First pass
PYRAMID_SCALE = 1.5
ROI_SIZE = (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
WINDOW_STEP = 8
NUM_OF_STEPS_SMALL_DIM = 8
DOWNSCALE_DIM = config.IMAGE_HEIGHT + NUM_OF_STEPS_SMALL_DIM*WINDOW_STEP
CONFIDENCE = 0.9

# Second pass
SECOND_PASS = 0
SECOND_PASS_WINDOW_STEP = 3
SECOND_PASS_NUM_OF_STEPS = int(config.BATCH_SIZE**(0.5))
SECOND_PASS_DOWNSCALE_DIM = config.IMAGE_HEIGHT + SECOND_PASS_NUM_OF_STEPS*SECOND_PASS_WINDOW_STEP

# ROI optimization
ROI_OPT = 1
OPT_KERNAL_SIZE = 5
OPT_DIFF_SHIFT = 2
OPT_COLOR_DIFF_THRESH = 30
OPT_SPATIAL_DIFF_FACTOR = 0.1
OPT_FRAME_BUFFER = 2

# Tracking
USE_TRACKING = 1
TRACKING_MAX_TRIES = 3
TRACKING_BUFFER_FACTOR = 1.25
TRACKING_BUFFER = (TRACKING_BUFFER_FACTOR - 1)/2
TRACKING_CONFIDENCE = 0.9

# Class contains tracking information 
class trackInfo:
	def __init__(self):
		self.detected = False
		self.x_size = 0
		self.y_size = 0
	
	def update(self, x, y, x_size, y_size):
		self.x = x
		self.y = y
		self.x_size = x_size
		self.y_size = y_size
	
	def get(self):
		return self.x, self.y, self.x_size, self.y_size

# Prepares image for detection, detects the balloons coordinates in a frame, calculates metrics, adds debug prints 
def detect_wrapper(frame, frame_idx, track_info):
	# Sample start time
	startTime = time.time()

	# Create temp folder for debug if needed
	if args["frame_debug"]:
		if not os.path.isdir("temp/frame{}".format(frame_idx)):
			os.mkdir("temp/frame{}".format(frame_idx))

	if args["frame_debug"]:
		cv2.imwrite("temp/frame{}/frame.jpg".format(frame_idx), frame)
	if args["debug"] >= 10:
		print("[DEBUG] Frame shape {}".format(frame.shape))

	if USE_TRACKING:
		if track_info.detected:
			(x, y, roi_size, detected) = track(frame, frame_idx, track_info)
			track_info.detected = detected
		else:
			(x, y, roi_size) = detect(frame, frame_idx)
			track_info.detected = True
	else:
		(x, y, roi_size) = detect(frame, frame_idx)
	
	# Optimize rectangle to fit balloons and add the rectangle to the frame
	if ROI_OPT:
		(x, y, x_size, y_size) = opt_rectangle(frame, x, y, roi_size, frame_idx, track_info)
	else:
		x_size = size
		y_size = size
	add_rectangle(frame, x, y, x_size, y_size, frame_idx)

	if args["debug"] >= 2:
		print("[DEBUG] Balloons detected in frame {} x {} y {} ROI size {}".format(frame_idx , x, y, roi_size))
	
	if args["frame_debug"]:
		cv2.imwrite("temp/frame{}/frame_out.jpg".format(frame_idx), frame)
		cv2.imwrite("temp/frame_out{}.jpg".format(frame_idx), frame)
	if args["show_out"]:
		cv2.imwrite("temp/frame_out{}.jpg".format(frame_idx), frame)
	if args["debug"] >= 10:
		print("[DEBUG] Frame shape {}".format(frame.shape))

	# Sample end time
	endTime = time.time()
	
	print("[INFO] Frame {} evaluated in {:.4f} seconds".format(frame_idx, endTime - startTime))
	
	track_info.update(x, y, x_size, y_size)
	return frame, (endTime - startTime), track_info

# Detects the balloons coordinates in a frame
def detect(frameLocal, frame_idx):
	# Init batch ROIs and coordinates
	batchROIs = None
	batchLocs = []

	# Downscale frame
	(h, w) = frameLocal.shape[:2]
	if w < h:
		# frameResized = cv2.resize(frameLocal, width=DOWNSCALE_DIM, interpolation=cv2.INTER_CUBIC)
		frameResized = imutils.resize(frameLocal, width=DOWNSCALE_DIM, inter=INTERPOLATION)
		# downScale = DOWNSCALE_DIM/w
		downScale = float(DOWNSCALE_DIM)/w
	else:
		# frameResized = cv2.resize(frameLocal, height=DOWNSCALE_DIM, interpolation=cv2.INTER_CUBIC)
		frameResized = imutils.resize(frameLocal, height=DOWNSCALE_DIM, inter=INTERPOLATION)
		# downScale = DOWNSCALE_DIM/h
		downScale = float(DOWNSCALE_DIM)/h

	if args["frame_debug"]:
		cv2.imwrite("temp/frame{}/frame_resized.jpg".format(frame_idx), frameResized)
	if args["debug"] >= 10:
		print("[DEBUG] Resized frame shape {}".format(frameResized.shape))
	
	# Init loop variables
	roi_idx = 0
	batch_idx = 0
	prob_best = 0
	x_best = 0
	y_best = 0
	idx_best = 0
	pyramid_best = 0
	done = False
	
	# Loop over the image pyramid
	for pyramid_idx, image in enumerate(image_pyramid(frameResized, scale=PYRAMID_SCALE, minSize=ROI_SIZE)):
		
		# Stop loop if done
		if done:
			break
	
		if args["debug"] >= 2:
			print("[DEBUG] Frame {} pyramid {} ROI {} shape {}".format(frame_idx, pyramid_idx, roi_idx, image.shape))
		
		# Loop over the sliding window locations
		for (x, y, roi) in sliding_window(image, WINDOW_STEP, ROI_SIZE):
			if args["frame_debug"]:
				cv2.imwrite("temp/frame{}/ROI{}.jpg".format(frame_idx, roi_idx), roi)
			
			# Pre-process ROI
			# roi = img_to_array(roi)
			# roi = np.expand_dims(roi, axis=0)
			# roi = imagenet_utils.preprocess_input(roi)
			if preprocessors is not None:
				for p in preprocessors:
					roi = p.preprocess(roi)
			roi = np.expand_dims(roi, axis=0)

			
			# Add ROI to batch
			if batchROIs is None:
				batchROIs = roi
			else:
				batchROIs = np.vstack([batchROIs, roi])

			# Add coordinates of the sliding window to the
			batchLocs.append((x, y, roi_idx, pyramid_idx))
			roi_idx += 1

			# Check if batch is full
			if len(batchROIs) == config.BATCH_SIZE:
				# Classify the batch
				predictStartTime = time.time()
				preds = model.predict(batchROIs)
				predictEndTime = time.time()
		
				# Update best prediction if better prediction was found
				prob_best, x_best, y_best, idx_best, pyramid_best = get_best_prediction(preds, batchLocs, prob_best, x_best, y_best, idx_best, pyramid_best)
			
				if args["debug"] >= 1:
					print("[DEBUG] Frame {} classified batch {} best balloon probability {:.4f} in ROI {} duration {:.4f} seconds"
						.format(frame_idx, batch_idx, prob_best, idx_best, predictEndTime - predictStartTime))

				# Reset batch of ROIs and coordinates
				batchROIs = None
				batchLocs = []
			
				batch_idx += 1
				
				# Stop loop if good enough result was found
				if prob_best > CONFIDENCE:
					done = True
					break

	# Construct a dummy ROI, first prediction is significantly longer in GPU
	# dummy_roi = np.zeros((config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3))
	# dummy_roi = np.expand_dims(dummy_roi, axis=0)

	# Check if there are any remaining ROIs to be classified
	if batchROIs is not None:
		# Extend the batch to be full size
		# for dummy_idx in range(0, (config.BATCH_SIZE - len(batchROIs))):
			# batchROIs = np.vstack([batchROIs, dummy_roi])
			# batchROIs = np.vstack([batchROIs, np.zeros((0, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3))])
			# batchLocs.append((0, 0, 0, 0))

		# Classify the batch
		predictStartTime = time.time()
		preds = model.predict(batchROIs)
		predictEndTime = time.time()
		
		# Update best prediction if better prediction was found
		prob_best, x_best, y_best, idx_best, pyramid_best = get_best_prediction(preds, batchLocs, prob_best, x_best, y_best, idx_best, pyramid_best)
	
		if args["debug"] >= 1:
			print("[DEBUG] Frame {} classified batch {} best balloon probability {:.4f} in ROI {} duration {:.4f} seconds batch size {}"
				.format(frame_idx, batch_idx, prob_best, idx_best, predictEndTime - predictStartTime, len(batchROIs)))
	
	if args["debug"] >= 1:
		print("[DEBUG] Frame {} classified {} ROI's best balloon probability {:.4f} in ROI {} pyramid {}".format(frame_idx, roi_idx, prob_best, idx_best, pyramid_best))
	
	
	# return x_best*downScale, y_best*downScale, config.IMAGE_HEIGHT*downScale
	
	downScale /= PYRAMID_SCALE**pyramid_best
	x_best = int(x_best/downScale)
	y_best = int(y_best/downScale)
	roi_size = int(config.IMAGE_HEIGHT/downScale)

	if SECOND_PASS:
		# Extract ROI which is most likely to contain balloons
		roi_best = frameLocal[y_best:y_best + roi_size, x_best:x_best + roi_size]
		
		if args["frame_debug"]:
			cv2.imwrite("temp/frame{}/ROI_best.jpg".format(frame_idx), roi_best)
		
		# Resize best ROI to fit into a single batch
		roi_best_resized = cv2.resize(roi_best, (SECOND_PASS_DOWNSCALE_DIM, SECOND_PASS_DOWNSCALE_DIM), interpolation=INTERPOLATION)
		second_pass_downScale = SECOND_PASS_DOWNSCALE_DIM/roi_size

		if args["frame_debug"]:
			cv2.imwrite("temp/frame{}/ROI_best_resized.jpg".format(frame_idx), roi_best_resized)
		
		# Reset batch of ROIs and coordinates
		batchROIs = None
		batchLocs = []
		
		# Init loop variables
		roi_idx = 0
		new_prob_best = 0
		new_x_best = 0
		new_y_best = 0
		new_idx_best = 0

		# Loop over the sliding window locations
		for (x, y, roi) in sliding_window(roi_best_resized, SECOND_PASS_WINDOW_STEP, ROI_SIZE):
			if args["frame_debug"]:
				cv2.imwrite("temp/frame{}/ROI_best{}.jpg".format(frame_idx, roi_idx), roi)
			
			# Pre-process ROI
			# roi = img_to_array(roi)
			# roi = np.expand_dims(roi, axis=0)
			# roi = imagenet_utils.preprocess_input(roi)
			if preprocessors is not None:
				for p in preprocessors:
					roi = p.preprocess(roi)
			roi = np.expand_dims(roi, axis=0)
			
			# Add ROI to batch
			if batchROIs is None:
				batchROIs = roi
			else:
				batchROIs = np.vstack([batchROIs, roi])

			# Add coordinates of the sliding window to the
			batchLocs.append((x, y, roi_idx, 0))

			roi_idx += 1

		# Check batch is full
		if len(batchROIs) != config.BATCH_SIZE:
			print("[WARNING] Batch is not full on second pass frame {} batch size {}".format(frame_idx, len(batchROIs)))

		# Classify the batch
		predictStartTime = time.time()
		preds = model.predict(batchROIs)
		predictEndTime = time.time()

		# Update best prediction if better prediction was found
		new_prob_best, new_x_best, new_y_best, new_idx_best, _ = get_best_prediction(preds, batchLocs)

		if args["debug"] >= 1:
			print("[DEBUG] Frame {} classified second pass best balloon probability {:.4f} in ROI {:3d} duration {:.4f} seconds"
				.format(frame_idx, new_prob_best, new_idx_best, predictEndTime - predictStartTime))

		if args["debug"] >= 8:
			print("[DEBUG] Frame {} classified second pass x_best {} new_x_best {} y_best {} new_y_best {} roi_size {} second_pass_downScale {:.4f}"
				.format(frame_idx, x_best, new_x_best, y_best, new_y_best, roi_size, second_pass_downScale))

		x_best += int(new_x_best/second_pass_downScale)
		y_best += int(new_y_best/second_pass_downScale)
		roi_size = int(config.IMAGE_HEIGHT/second_pass_downScale)

	return x_best, y_best, roi_size

# Track the balloons in the frame
def track(frameLocal, frame_idx, track_info):
	# Init variables
	roi_idx = 0
	done = False
	
	# Get ROI location
	x, y, x_size, y_size = track_info.get()
	
	# Update ROI to be square
	if x_size < y_size:
		roi_size = y_size
		x -= int((y_size-x_size)/2)
	else:
		roi_size = x_size
		y -= int((y_size-x_size)/2)

	# Loop until detection is successful
	while not done and roi_idx < TRACKING_MAX_TRIES:
		# Extract ROI with additional buffer
		x -= int(roi_size*TRACKING_BUFFER)
		y -= int(roi_size*TRACKING_BUFFER)
		roi_size += int(roi_size*TRACKING_BUFFER*2)
		roi = frameLocal[y:y + roi_size, x:x + roi_size]

		if args["frame_debug"]:
			cv2.imwrite("temp/frame{}/ROI{}.jpg".format(frame_idx, roi_idx), roi)

		# Pre-process ROI
		if preprocessors is not None:
			for p in preprocessors:
				roi = p.preprocess(roi)
		roi = np.expand_dims(roi, axis=0)
		
		# Classify the batch
		predictStartTime = time.time()
		preds = model.predict(roi)
		predictEndTime = time.time()
		
		# Extract probability and prediction
		prob = preds[0,0]
		label = preds[0,1].astype(int)
		
		if args["debug"] >= 1:
			print("[DEBUG] Frame {} classified ROI {} balloon probability {:.4f} duration {:.4f} seconds"
				.format(frame_idx, roi_idx, prob, predictEndTime - predictStartTime))
		
		# Stop loop if good enough result was found
		if label == BALLOONS_LABLE and prob > TRACKING_CONFIDENCE:
			done = True
			break

		roi_idx += 1
		
	return x, y, roi_size, done

# Updates the best prediction info if new batch has better prediction
def get_best_prediction(preds, batchLocs, prob_best=0, x_best=0, y_best=0, idx_best=0, pyramid_best=0):
	# Extract probabilities of balloon prediction in new batch 
	probs = preds[:,0]
	labels = preds[:,1].astype(int)
	if (BALLOONS_LABLE == 0):
		labels = 1 - labels
	probs = probs * labels
	
	# Find best prediction info if new batch
	argMax = np.argmax(probs)
	prob_best_new = probs[argMax]
	x_best_new, y_best_new, idx_best_new, pyramid_best_new = batchLocs[argMax]
	
	# Update prediction info if it is better than best prediction
	if (prob_best_new > prob_best):
		prob_best = prob_best_new
		x_best = x_best_new
		y_best = y_best_new
		idx_best = idx_best_new
		pyramid_best = pyramid_best_new
	
	return prob_best, x_best, y_best, idx_best, pyramid_best

# Add rectangle to a frame
def opt_rectangle(frameLocal, x_local, y_local, size, frame_idx, track_info):
	# Extract ROI
	roi = frameLocal[y_local:y_local + size, x_local:x_local + size]
	
	# Smooth ROI
	kernel = np.ones((OPT_KERNAL_SIZE, OPT_KERNAL_SIZE), np.float32)/(OPT_KERNAL_SIZE**2)
	roi = cv2.filter2D(roi, -1, kernel)
	
	# Construct diff arrays
	roi = roi.astype(int)
	roi_diff_vertical   = abs(roi[OPT_DIFF_SHIFT:,:,:] - roi[:-OPT_DIFF_SHIFT,:,:])
	roi_diff_horizontal = abs(roi[:,OPT_DIFF_SHIFT:,:] - roi[:,:-OPT_DIFF_SHIFT,:])
	
	# Calculate spatial diff thresholds
	if track_info.y_size:
		vertical_diff_thresh = OPT_SPATIAL_DIFF_FACTOR*track_info.y_size
	else:
		vertical_diff_thresh = OPT_SPATIAL_DIFF_FACTOR*roi.shape[0]
	if track_info.x_size:
		horizontal_diff_thresh = OPT_SPATIAL_DIFF_FACTOR*track_info.x_size
	else:
		horizontal_diff_thresh = OPT_SPATIAL_DIFF_FACTOR*roi.shape[1]
	
	# Get upper and lower balloons location
	roi_diff_vertical[roi_diff_vertical <  OPT_COLOR_DIFF_THRESH] = 0
	roi_diff_vertical[roi_diff_vertical >= OPT_COLOR_DIFF_THRESH] = 1
	roi_vertical_sum = np.sum(roi_diff_vertical, (1, 2))
	
	# print(roi_vertical_sum)
	
	roi_vertical_sum[roi_vertical_sum < vertical_diff_thresh] = 0
	vertical_idx_list = np.nonzero(roi_vertical_sum)
	
	# print(roi_diff_vertical.shape)
	# print(roi_vertical_sum)
	
	# Get left and right balloons location
	roi_diff_horizontal[roi_diff_horizontal <  OPT_COLOR_DIFF_THRESH] = 0
	roi_diff_horizontal[roi_diff_horizontal >= OPT_COLOR_DIFF_THRESH] = 1
	roi_horizontal_sum = np.sum(roi_diff_horizontal, (0, 2))
	
	# print(roi_horizontal_sum)
	
	roi_horizontal_sum[roi_horizontal_sum < horizontal_diff_thresh] = 0
	horizontal_idx_list = np.nonzero(roi_horizontal_sum)
	
	# print(roi_diff_horizontal.shape)
	# print(roi_horizontal_sum)
	
	# Update ROI location
	# print (len(horizontal_idx_list[0]))
	# print (len(vertical_idx_list[0]))
	if len(horizontal_idx_list[0]) and len(vertical_idx_list[0]):
		x_best = x_local + horizontal_idx_list[0][0] - OPT_FRAME_BUFFER
		y_best = y_local + vertical_idx_list[0][0]   - OPT_FRAME_BUFFER
		x_size = horizontal_idx_list[0][-1] - horizontal_idx_list[0][0] + 2*OPT_FRAME_BUFFER + OPT_DIFF_SHIFT
		y_size = vertical_idx_list[0][-1]   - vertical_idx_list[0][0]   + 2*OPT_FRAME_BUFFER + OPT_DIFF_SHIFT
	else:
		x_best, y_best = x_local, y_local
		x_size, y_size = size, size
	
	if args["debug"] >= 8:
		print("[DEBUG] Frame {} initial ROI location x {} y {} x_size {} y_size {}".format(frame_idx, x_local, y_local, size, size))
		print("[DEBUG] Frame {} optimized ROI location x {} y {} x_size {} y_size {}".format(frame_idx, x_best, y_best, x_size, y_size))
	
	return x_best, y_best, x_size, y_size

# Add rectangle to a frame
def add_rectangle(frameLocal, x_local, y_local, x_size, y_size, frame_idx):

	# Draw a green rectangle around the ROI
	color = (0, 255, 0)
	refPointsOpt = []
	refPointsOpt.append((x_local       , y_local     ))
	refPointsOpt.append((x_local+x_size, y_local+y_size))
	cv2.rectangle(frameLocal, refPointsOpt[0], refPointsOpt[1], color, 3)

	if args["frame_number"]:
		color = (0, 0, 255)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frameLocal, str(frame_idx), (30, 70), font, 2, color, 3)

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video file or directory of images")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained CNN")
ap.add_argument("-sf", "--start_frame", type=int, default=0,
	help="frame number from which cropping will start")
ap.add_argument("-f", "--frame", type=int, default=0,
	help="frame number to run, only this frame will run and this will override any other settings")
ap.add_argument("-d", "--debug", type=int, default=0,
	help="enable debug option")
ap.add_argument("-so", "--show_out", type=int, default=0,
	help="write result to temp file during detection")
ap.add_argument("-fd", "--frame_debug", type=int, default=0,
	help="write results and debug images to temp file during detection")
ap.add_argument("-fn", "--frame_number", type=int, default=0,
	help="add frame number to the video")
args = vars(ap.parse_args())

if not os.path.isdir(args["input"]): # Video input
	# Parse input video name
	videoName = args["input"]
	videoName = videoName.split("/")[-1]
	videoName = videoName.split(".")[-2]

	# Parse output video name and print
	pathToDir = args["input"]
	pathToDir = pathToDir.rsplit("/", 1)[0]
	outputPath = os.path.sep.join([pathToDir, "{}_with_balloons.{}".format(videoName, FORMAT)])
	print("[INFO] Output video is {}".format(outputPath))

	# Create a VideoCapture object to read from input video file
	cap = cv2.VideoCapture(args["input"])

	# Check if successful
	if (cap.isOpened()== False): 
		print("Error opening video file")
	else:
		if args["debug"] == 8:
			print("[DEBUG] Video has {} frames in {} fps".format(
				int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FPS))))
		if args["debug"] == 10:
			print("[DEBUG] Video width {} height {}".format(
				int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))

	# Create a VideoWriter object to write into output video file
	fourcc = cv2.VideoWriter_fourcc(*FOURCC)
	dims = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
	out = cv2.VideoWriter(outputPath, fourcc, cap.get(cv2.CAP_PROP_FPS), dims)
else: # Directory input
	
	# Grab images paths
	imagePaths = list(paths.list_images(args["input"]))
	
	# sorted(imagePaths, key=lambda x: (x.split(os.path.sep)[-1], x.split('.')[0], x = int(x)))
	
	# for path in imagePaths:
		# print(path)
		# path = path.split(os.path.sep)[-1]
		# path = path.split('.')[0]
		# print(path)

	# Create output folder 
	outputDir = args["input"]
	outputDir = "{}_with_balloons".format(outputDir)
	if not os.path.isdir(outputDir):
		os.mkdir(outputDir)
		
	print("[INFO] Output directory is {}".format(outputDir))
		
# Create temp folder for debug if needed
if args["frame_debug"] | args["show_out"]:
	if not os.path.isdir("temp"):
		os.mkdir("temp")

# Load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# Init preprocessors
sp = SimplePreprocessor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()
preprocessors=[sp, mp, iap]

# Load the network weights
print("[INFO] Loading network")
model = load_model(args["model"])

# Predict on dummy batch, first prediction is significantly longer in GPU
dummy_batch  = np.zeros((config.BATCH_SIZE, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3))
_ = model.predict(dummy_batch)
dummy_batch  = np.zeros((1, config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3))
_ = model.predict(dummy_batch)

# Init video loop variables
frame_idx = 0
total_frame_time = 0
track_info = trackInfo()
if args["frame"]:
	NUM_OF_FRAMES = args["frame"] + 1

if not os.path.isdir(args["input"]):
	# Skip frames when not starting from first frame
	while (cap.isOpened() & ((frame_idx < args["start_frame"]) | (frame_idx < args["frame"]))):
		success, frame = cap.read()
		frame_idx+=1

	print("[INFO] Processing video")

	# Read until video is completed
	while (cap.isOpened() & ((NUM_OF_FRAMES == 0) | (frame_idx < NUM_OF_FRAMES))):
		# Read frame
		success, frame = cap.read()
		if success == True:
			# Rotate frame 90 degrees clockwise
			frame = np.rot90(frame, 3, (0,1))
			frame = frame.copy() # Solve rot90 and add_rectangle OpenCV bug

			# Detect balloons in the frame
			frame, frame_time, track_info = detect_wrapper(frame, frame_idx, track_info)
			frame_idx += 1
			total_frame_time += frame_time

			# Write frame
			out.write(frame)
		else: 
			break
else:
	print("[INFO] Processing directory")
	
	# Read entire directory
	# for path in imagePaths:
	while (frame_idx < len(imagePaths)):
		path = os.path.sep.join([args["input"], "{}.jpg".format(frame_idx)])
		# Read frame
		frame = cv2.imread(path)
		
		# Detect balloons in the frame
		frame, frame_time, track_info = detect_wrapper(frame, frame_idx, track_info)
		frame_idx += 1
		total_frame_time += frame_time
		
		# Write frame
		outputPath = os.path.sep.join([outputDir, "{}.jpg".format(frame_idx)])
		cv2.imwrite(outputPath, frame)
		
		if (frame_idx == NUM_OF_FRAMES):
			break

print("[INFO] Evaluation rate is {:.3f} fps".format(frame_idx/total_frame_time))

if not os.path.isdir(args["input"]):
	# Release the VideoCapture object
	cap.release()

	# Release the VideoWriter object
	out.release()

	# Closes all open frames
	cv2.destroyAllWindows()
