import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))