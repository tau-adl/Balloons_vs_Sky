# Import the necessary packages

def sliding_window(image, step, ws):
	# Slide a window across the image
	for x in range(0, image.shape[1] - ws[0], step):
		for y in range(0, image.shape[0] - ws[1], step):
			# Yield the current window
			yield (x, y, image[y:y + ws[1], x:x + ws[0]])
