import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


MAX_CAPTION_LENGTH = 30
FREQ_THRESHOLD = 2

TRAIN_SPLIT = 0.8



EMBED_SIZE = 256
HIDDEN_SIZE = 512
ATTENTION_DIM = 256
NUM_LAYERS = 1
DROPOUT = 0.3


BATCH_SIZE = 16
LEARNING_RATE = 3e-4
NUM_EPOCHS = 12

DEVICE = "cpu" # Force CPU for your machine
