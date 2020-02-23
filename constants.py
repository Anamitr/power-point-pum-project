# images
BASE_PATH = './db3/'
BASE_IMAGE_EXTENSION = '.jpg'
# TYPES_OF_GESTURES = ['close', 'left', 'open', 'play', 'right']
TYPES_OF_GESTURES = ['close', 'play2', 'left', 'right', 'open', 'volume_up', 'volume_down2', 'pointer2']
PROJECT_NAME = 'PUM hand recognition'
TRAINED_MODELS_FOLDER = 'trained_models'

# recognition
TRAINING_PART_RATIO = 0.2
CLASSIFICATION_PROBABILITY_THRESHOLD = 0.60
SIGN_REPETITION_THRESHOLD = 10
POINTER_SIGN_REPETITION_THRESHOLD = 3

# screen resolution
MIN_X = 2934
MIN_Y = 10
MAX_X = 4623
MAX_Y = 1068
SCREEN_WIDTH = MAX_X - MIN_X
SCREEN_HEIGHT = MAX_Y - MIN_Y
