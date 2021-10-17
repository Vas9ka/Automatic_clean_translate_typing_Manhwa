import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("wastedata-Mask_RCNN-multiple-classes/main/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn.config import Config
from tensorflow.python.keras.backend import set_session
from keras.backend import get_session

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WEIGHTS_PATH = "cloud_detection_model/v2.h5"

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
# Inspect the model in training or inference modes values: 'inference' or 'training'
TEST_MODE = "inference"


class CustomConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + (Horse and Man)
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


config = CustomConfig()


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.90


config = InferenceConfig()
config.display()


class CloudDetModel:
    def __init__(self):
        self.session = get_session()
        init = tf.global_variables_initializer()
        self.session.run(init)
        self.session.run(init)
        DEVICE = "/gpu:0"
        with tf.device(DEVICE):
            self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        weights_path = WEIGHTS_PATH
        print("Loading weights ", weights_path)
        self.model.load_weights(weights_path, by_name=True)
        self.graph = tf.get_default_graph()
        set_session(self.session)

    def pred(self, image):
        with self.session.as_default():
            with self.graph.as_default():
                set_session(self.session)
                result = self.model.detect([image], verbose=1)
        return result
