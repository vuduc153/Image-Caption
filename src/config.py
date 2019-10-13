import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DATA_PATH = os.path.join(BASE_DIR, 'Flicker8k_Dataset')
DESC_DATA_PATH = os.path.join(BASE_DIR, 'Flickr8k_text/Flickr8k.lemma.token.txt')
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'Flickr8k_text/Flickr_8k.trainImages.txt')
VECTORIZED_DATA_DIR = os.path.join(BASE_DIR, 'src/vectorized_data')
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')

MAX_NUM_WORDS = 100000
EMBEDDING_DIM = 200
BATCH_SIZE = 3

NUM_TRAIN_IMAGES = 6000

MODEL_DIR = os.path.join(BASE_DIR, 'src/model')
