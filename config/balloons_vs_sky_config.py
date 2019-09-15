# Path to images directory
IMAGES_PATH = "datasets/balloons_vs_sky_basic"

# Set dataset_name
DATASET_NAME = "balloons_vs_sky_basic"

# Define dataset parameters
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CLASSES = 2
BATCH_SIZE = 64
VAL_IMAGES_PCT   = .10
TEST_IMAGES_PCT  = .10

# Path to the output training, validation, and testing HDF5 files
TRAIN_HDF5 = "datasets/balloons_vs_sky_basic/hdf5/train.hdf5"
VAL_HDF5   = "datasets/balloons_vs_sky_basic/hdf5/val.hdf5"
TEST_HDF5  = "datasets/balloons_vs_sky_basic/hdf5/test.hdf5"

# Path to the dataset mean
DATASET_MEAN = "datasets/balloons_vs_sky_basic/balloons_vs_sky_basic_mean.json"

# Path to the output directory
OUTPUT_PATH = "output/balloons_vs_sky_basic"