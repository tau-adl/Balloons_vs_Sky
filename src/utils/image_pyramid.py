# Import the necessary packages
import imutils

def image_pyramid(image, scale=1.2, minSize=(64, 64)):
	# Yield the original image
	yield image

	# Loop over the image pyramid
	while True:
		# Compute the dimensions of the next image in the pyramid and resize accordingly
		width = int(image.shape[1]/scale)
		image = imutils.resize(image, width=width)

		# Stop if the resized image does not meet the supplied minimum size
		if (image.shape[0] < minSize[0]) or (image.shape[1] < minSize[1]):
			break

		# Yield the next image in the pyramid
		yield image
