# Usage
# python crop_balloons_from_video.py --input datasets/balloons_vs_sky_basic/train/20190320_162139.mp4 --start_frame 10

# Import packages
import numpy as np
import argparse
import cv2
import os

NUM_OF_FRAMES = 0 # Set to zero to run on entire video

# Automatic mode
BORDER_WIDTH = 5
NEXT_FRAME_MAX_CHANGE = 15
DIFF_THRESH = 70

# Manual mode
NUM_OF_PIXEL_TO_MOVE = 5

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-sf", "--start_frame", type=int, default=0,
	help="frame number from which cropping will start")
ap.add_argument("-m", "--auto", type=int, default=0,
	help="crop frames automatically")
ap.add_argument("-d", "--debug", type=int, default=0,
	help="enable debug option")
args = vars(ap.parse_args())

# Initialize the list of background RGB means
skyMeans = []

# Initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPoints = []
cropping = False
x1 = 0
x2 = 0
y1 = 0
y2 = 0
sky_direction = 0

# Record the rectangle dimensions when cropping with the mouse and display result
def crop_frame(event, x, y, flags, param):
	# Grab references to the global variables
	global refPoints, cropping

	# If the left mouse button was clicked
	if event == cv2.EVENT_LBUTTONDOWN:
		# Record the starting (x, y) coordinates and indicate that
		# cropping is being performed
		refPoints = [(x, y)]
		cropping = True

	# If the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# Record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPoints.append((x, y))
		cropping = False

		# Draw a rectangle around the ROI
		color = (0, 255, 0)
		cv2.rectangle(frame, refPoints[0], refPoints[1], color, 3)
		cv2.imshow("frame", frame)

# Transform the ROI rectangle captured into a square
def square_ROI():
	# Grab references to the global variables
	global x1 ,x2, y1, y2

	# Extract corners
	x1 = refPoints[0][0]
	x2 = refPoints[1][0]
	y1 = refPoints[0][1]
	y2 = refPoints[1][1]
	
	# Extract height and width, make equal 
	height = y2 - y1
	width  = x2 - x1
	if height > width:
		diff = (height - width)//2
		x1 -= diff
		x2 += diff
	else:
		diff = (width - height)//2
		y1 -= diff
		y2 += diff

# Display a rectangle around the ROI
def show_ROI(frameLocal, show=False):
	# Grab references to the global variables
	global x1 ,x2, y1, y2

	# Draw a rectangle around the ROI
	color = (0, 0, 255)
	refPointsOpt = []
	refPointsOpt.append((x1, y1))
	refPointsOpt.append((x2, y2))
	cv2.rectangle(frameLocal, refPointsOpt[0], refPointsOpt[1], color, 3)
	if show:
		cv2.imshow("frame", frameLocal)

# Calculate the RGB means of the sky background
# Save for future reference if in init phase
# return False if means are different than reference values not in init phase
def calc_sky_means(frameLocal):
	global skyMeans

	if args["debug"]:
		print("[DEBUG] Frame shape {} x1 {} x2 {} y1 {} y2 {}".format(frameLocal.shape, x1, x2, y1, y2))

	# Construct list of border pixels for calculation
	borderList = []
	for idx in range (-BORDER_WIDTH//2 + 1, BORDER_WIDTH//2 + 1):
		if args["debug"]:
			print("[DEBUG] Border list loop {:+d}".format(idx))
		borderList.extend(frameLocal[y1+idx, x1:x2 ]) # Top
		borderList.extend(frameLocal[y2+idx, x1:x2 ]) # Bottom
		borderList.extend(frameLocal[y1:y2 , x1+idx]) # Left
		borderList.extend(frameLocal[y1:y2 , x2+idx]) # Right

	if args["debug"]:
		print("[DEBUG] Border list len {}".format(len(borderList)))
	
	# Save channel averages of the border
	skyMeans = np.mean(borderList, axis = 0)
	print("[INFO] Sky means {}".format(skyMeans))

# Returns the pixel percentage for a given ROI
def calc_non_sky_pixels(frameLocal, x1_local, x2_local, y1_local, y2_local):
	global skyMeans

	if args["debug"]:
		print("[DEBUG] Non sky pixel start x1 {} x2 {} y1 {} y2 {}".format(x1_local, x2_local, y1_local, y2_local))

	# Extract the ROI
	frameROI = frameLocal[y1_local:y2_local,x1_local:x2_local]
	if args["debug"]:
		print("[DEBUG] Frame ROI shape {}".format(frameROI.shape))
		cv2.imshow("Frame ROI", frameROI)
	
	# Create an image array of the background values with the same size as the ROI
	skyROI = np.tile(skyMeans, [frameROI.shape[0], frameROI.shape[1], 1])
	if args["debug"]:
		print("[DEBUG] Sky ROI shape {}".format(skyROI.shape))
		cv2.imshow("Sky ROI", skyROI.astype(np.uint8))

	# Calculate L2 distance
	diff = frameROI - skyROI
	dist = np.sqrt(np.sum(diff**2, axis=2))
	if args["debug"]:
		print("[DEBUG] Dist shape {}".format(dist.shape))
		cv2.imshow("Dist ROI", dist.astype(np.uint8))

	# Threshold the result
	ret, thresh = cv2.threshold(dist, DIFF_THRESH, 1, cv2.THRESH_BINARY)

	# Calculate object pixel percentage
	# if thresh.any():
	if (thresh is not None):
		pixelCount = sum(sum(thresh))
		pixelPct = pixelCount/(frameROI.shape[0]*frameROI.shape[1])
		if args["debug"]:
			print("[DEBUG] pixelCount {}".format(pixelCount))
			print("[DEBUG] pixelPct {}".format(pixelPct))
		return pixelPct
	else:
		return 0;

# Updates the ROI coordinates with the maximum pixel percentage
def update_max_pixels_ROI(frameLocal):
	# Grab references to the global variables
	global x1 ,x2, y1, y2

	if args["debug"]:
		print("[DEBUG] Max pixel start x1 {} x2 {} y1 {} y2 {}".format(x1, x2, y1, y2))
	
	# Create an array of pixel percentages for a shifted frame around current frame
	pixelPctArray = np.zeros((2*NEXT_FRAME_MAX_CHANGE+1, 2*NEXT_FRAME_MAX_CHANGE+1))
	for x_idx in range (0, 2*NEXT_FRAME_MAX_CHANGE+1):
		for y_idx in range (0, 2*NEXT_FRAME_MAX_CHANGE+1):
			x_shift = x_idx - NEXT_FRAME_MAX_CHANGE
			y_shift = y_idx - NEXT_FRAME_MAX_CHANGE
			if args["debug"]:
				print("[DEBUG] Max pixel loop x {} y {} x_shift {:+d} y_shift {:+d}".format(x_idx, y_idx, x_shift, y_shift))
			
			pixelPctArray[x_idx][y_idx] = calc_non_sky_pixels(frameLocal, x1+x_shift, x2+x_shift, y1+y_shift, y2+y_shift)
	
	if args["debug"]:
		print("[DEBUG] Max pixel array {}".format(pixelPctArray))

	# Calculate index shift for maximum pixel percentage
	x_max, y_max = np.unravel_index(pixelPctArray.argmax(), pixelPctArray.shape)
	
	# Update global frame location values
	x1 = x1 + x_max - NEXT_FRAME_MAX_CHANGE
	x2 = x2 + x_max - NEXT_FRAME_MAX_CHANGE
	y1 = y1 + y_max - NEXT_FRAME_MAX_CHANGE
	y2 = y2 + y_max - NEXT_FRAME_MAX_CHANGE
	
	if args["debug"]:
		print("[DEBUG] Max pixel done x1 {} x2 {} y1 {} y2 {}".format(x1, x2, y1, y2))

# Returns a sky image from one of the 4 neighbors images of the ROI
def show_sky_ROI(frameLocal, show=False):
	# Grab references to the global variables
	global x1 ,x2, y1, y2, sky_direction
	
	# Create new shifted coordinates to extract a sky image
	x1_shift = x1
	x2_shift = x2
	y1_shift = y1
	y2_shift = y2
	
	if args["debug"]:
		print("[DEBUG] Frame shape {} x1 {} x2 {} y1 {} y2 {}".format(frameLocal.shape, x1_shift, x2_shift, y1_shift, y2_shift))
	
	# Calculate shift: the size of the ROI (square) and some additional buffer
	if args["auto"]:
		shift = x2-x1+10
	else:
		shift = x2-x1-30
	
	if   (sky_direction == 0): # Up
		y1_shift -= shift
		y2_shift -= shift
	elif (sky_direction == 1): # Down
		y1_shift += shift
		y2_shift += shift
	elif (sky_direction == 2): # Left
		x1_shift -= shift
		x2_shift -= shift
	elif (sky_direction == 3): # Right
		x1_shift += shift
		x2_shift += shift

	if args["debug"]:
		print("[DEBUG] Frame shape {} x1 {} x2 {} y1 {} y2 {}".format(frameLocal.shape, x1_shift, x2_shift, y1_shift, y2_shift))

	if show:
		# Draw a rectangle around the ROI
		color = (255, 0, 0)
		refPointsOpt = []
		refPointsOpt.append((x1_shift, y1_shift))
		refPointsOpt.append((x2_shift, y2_shift))
		cv2.rectangle(frameLocal, refPointsOpt[0], refPointsOpt[1], color, 1)
	
		cv2.imshow("frame", frameLocal)

	return frameLocal[y1_shift:y2_shift,x1_shift:x2_shift,:]

# Parse video name and print
videoName = args["input"]
videoName = videoName.split("/")[-1]
videoName = videoName.split(".")[-2]
print("[INFO] Processing video {}".format(videoName))

# Parse output directory and print
pathToDir = args["input"]
pathToDir = pathToDir.rsplit("/", 1)[0]
outputDir = os.path.sep.join([pathToDir, videoName])
print("[INFO] Output directory {}".format(outputDir))

# Create output directories
if not os.path.isdir(outputDir):
	os.mkdir(outputDir)
	os.mkdir(os.path.sep.join([outputDir, "balloons"]))
	os.mkdir(os.path.sep.join([outputDir, "origin"]))
	os.mkdir(os.path.sep.join([outputDir, "sky"]))

# Create a VideoCapture object to read from input video file
cap = cv2.VideoCapture(args["input"])

# Check if successful
if (cap.isOpened()== False): 
	print("Error opening video file")
else:
	print("[INFO] Video has {} frames".format(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

# Init variables
done = False
quit = 0
first_frame = 1
frame_idx = 0

# Skip frames when not starting from first frame
while (cap.isOpened() & (frame_idx < args["start_frame"])):
	success, frame = cap.read()
	frame_idx+=1

# Read until video is completed
while (cap.isOpened() & (not done) & ((NUM_OF_FRAMES == 0) | (frame_idx < NUM_OF_FRAMES))):
	if quit:
		exit()
	
	# Read frame
	success, frame = cap.read()
	skip = False
	if success == True:
		# Clone frame
		frameClone = frame.copy()

		# Generate a random number between 0 and 3 for sky direction
		# sky_direction = np.random.randint(4)

		if first_frame:
			first_frame = 0

			while True:
				# Set callback
				cv2.namedWindow("frame")
				cv2.setMouseCallback("frame", crop_frame)
			
				# Display the frame and wait for a key-press
				cv2.imshow('frame', frame)
				key = cv2.waitKey(1) & 0xFF

				# if the 'r' key is pressed, reset the cropping region
				if key == ord("r"):
					print("[INFO] Retrying")
					frame = frameClone.copy()

				# if the 'o' key is pressed, start automatic cropping
				if key == ord("o"):
					cv2.destroyAllWindows()
					square_ROI()
					show_ROI(frame, show=True)
					show_sky_ROI(frame, show=True)
					if args["auto"]:
						calc_sky_means(frameClone.copy())
						pixelPct = calc_non_sky_pixels(frameClone.copy(), x1, x2, y1, y2)
						print("[INFO] Frame {} {:.2f} of pixels in ROI are balloons".format(frame_idx, pixelPct))
				
				# if the 'd' key is pressed, start automatic cropping
				if key == ord("d"):
					cv2.destroyAllWindows()
					break
				
				# if the 's' key is pressed, choose another sky ROI
				if key == ord("s"):
					sky_direction += 1
					if sky_direction == 4:
						sky_direction = 0
					show_sky_ROI(frame, show=True)
				
				# if the 'q' key is pressed, exit
				if key == ord("q"):
					quit = 1;
					exit()
		else:
			if args["auto"]:
				update_max_pixels_ROI(frameClone.copy())
				pixelPct = calc_non_sky_pixels(frameClone.copy(), x1, x2, y1, y2)
				print("[INFO] Frame {} {:.2f} of pixels in ROI are balloons".format(frame_idx, pixelPct))
			else:
				while True:
					# Display the frame and wait for a key-press
					frame = frameClone.copy()
					show_ROI(frame, show=True)
					show_sky_ROI(frame, show=True)
					key = cv2.waitKey(1) & 0xFF
					
					# Move rectangle according to number pad keys
					if   key == ord("5"):
						# cv2.destroyAllWindows()
						break
					elif key == ord("8"): # Up
							y1-=NUM_OF_PIXEL_TO_MOVE
							y2-=NUM_OF_PIXEL_TO_MOVE
							# cv2.destroyAllWindows()
							# continue
					elif key == ord("2"): # Down
							y1+=NUM_OF_PIXEL_TO_MOVE
							y2+=NUM_OF_PIXEL_TO_MOVE
							# cv2.destroyAllWindows()
							# continue
					elif key == ord("4"): # Left
							x1-=NUM_OF_PIXEL_TO_MOVE
							x2-=NUM_OF_PIXEL_TO_MOVE
							# cv2.destroyAllWindows()
							# continue
					elif key == ord("6"): # Right
							x1+=NUM_OF_PIXEL_TO_MOVE
							x2+=NUM_OF_PIXEL_TO_MOVE
							# cv2.destroyAllWindows()
							# continue
			
					# if the 's' key is pressed, choose another sky ROI
					if key == ord("s"):
						sky_direction += 1
						if sky_direction == 4:
							sky_direction = 0
					
					if key == ord("d"):
						skip = True
						break
					
					# if the 'q' key is pressed, exit
					if key == ord("q"):
						quit = 1;
						exit()	
		
		if not skip:
			# Write the ROI image to output dir
			frameROI = frameClone[y1:y2,x1:x2]
			outputPath = os.path.sep.join([outputDir, "balloons", "{}.jpg".format(frame_idx)])
			cv2.imwrite(outputPath, frameROI)
			
			# Write the frame with highlight of the ROI
			show_ROI(frame)
			outputPath = os.path.sep.join([outputDir, "origin", "{}.jpg".format(frame_idx)])
			cv2.imwrite(outputPath, frame)

			# Write a sky image
			skyROI = show_sky_ROI(frameClone.copy())
			outputPath = os.path.sep.join([outputDir, "sky", "{}.jpg".format(frame_idx)])
			cv2.imwrite(outputPath, skyROI)

		if args["auto"]:
			# Stop if ROI doesn't converge
			if (pixelPct > 0.9):
				done = True
				print("[INFO] Stop due to ROI is no longer converging in frame {}".format(frame_idx))
			
			# Stop if zooming out
			if (pixelPct < 0.3):
				done = True
				print("[INFO] Stop due to ROI is zoomed out in frame {}".format(frame_idx))

		frame_idx+=1
	else: 
		break

# Release the VideoCapture object
cap.release()

# Closes all open frames
cv2.destroyAllWindows()
