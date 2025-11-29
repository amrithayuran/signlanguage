import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

DATA2_DIR = os.path.join(BASE_DIR, 'data2')
TRAIN2_DIR = os.path.join(DATA2_DIR, 'train')
TEST2_DIR = os.path.join(DATA2_DIR, 'test')

# Model Directory
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_BW_JSON = os.path.join(MODEL_DIR, 'model-bw.json')
MODEL_BW_H5 = os.path.join(MODEL_DIR, 'model-bw.h5')
MODEL_BW_DRU_JSON = os.path.join(MODEL_DIR, 'model-bw_dru.json')
MODEL_BW_DRU_H5 = os.path.join(MODEL_DIR, 'model-bw_dru.h5')

# Image Processing
IMG_SIZE = 128  # Size for model input
ROI_SIZE = 300  # Size for Region of Interest in UI/Collection
MIN_VALUE = 70  # Threshold value

# Training Hyperparameters
BATCH_SIZE = 10
EPOCHS = 5
TRAIN_STEPS = 1000 # Adjusted from hardcoded 12841 for testing, should be dynamic
VAL_STEPS = 100    # Adjusted from hardcoded 4268

# UI Settings
WINDOW_TITLE = "Sign Language to Text Translator"
THEME_COLOR = "#FFC0CB"    # Pink
ACCENT_COLOR = "#FF1493"   # DeepPink
PANEL_BG_COLOR = "#FFB6C1" # LightPink
BUTTON_COLOR = "#FF69B4"   # HotPink
TEXT_COLOR = "#000000"     # Black text for better contrast on pink
