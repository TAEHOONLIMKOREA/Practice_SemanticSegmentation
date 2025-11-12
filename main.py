from TrainMethods import data_generator, read_image, train
from InferMethods import plot_predictions

from scipy.io import loadmat
import numpy as np
import os
from glob import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import json


NUM_CLASSES = 20
DATA_DIR = "./instance-level_human_parsing/instance-level_human_parsing/Training"
NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 50


def main():
    # # GPU 동작 확인
    # print("TensorFlow version:", tf.__version__)
    # print("Available Physical GPUs:")
    # print(tf.config.list_physical_devices('GPU'))

    # # 디바이스 배치 로깅 (옵션)
    # tf.debugging.set_log_device_placement(True)

    # # 혹은 MirroredStrategy에서 실제로 몇 개 GPU를 썼는지 확인
    # strategy = tf.distribute.MirroredStrategy()
    # print("Number of devices in strategy:", strategy.num_replicas_in_sync)
    
    # [1] 훈련용 데이터셋 준비
    train_images = sorted(glob(os.path.join(DATA_DIR, 'Images/*')))[:NUM_TRAIN_IMAGES]
    train_masks = sorted(glob(os.path.join(DATA_DIR, 'Category_ids/*')))[:NUM_TRAIN_IMAGES]
    val_images = sorted(glob(os.path.join(DATA_DIR, 'Images/*')))[NUM_TRAIN_IMAGES:NUM_VAL_IMAGES+NUM_TRAIN_IMAGES]
    val_masks = sorted(glob(os.path.join(DATA_DIR, 'Category_ids/*')))[NUM_TRAIN_IMAGES:NUM_VAL_IMAGES+NUM_TRAIN_IMAGES]

    train_dataset = data_generator(train_images, train_masks)
    val_dataset = data_generator(val_images, val_masks)    
    
    # [2-1] 학습  
    model = train(train_dataset, val_dataset)
    
    # # [2-2] 모델 불러오기
    # model = keras.models.load_model('my_model.h5')

    # [3] 추론 
    # 데이터셋과 함께 제공된 human_colormap.mat 파일을 통해 각 라벨에 대한 해당 색상을 찾을 수 있음
    colormap = loadmat('./instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat')['colormap']
    colormap = colormap * 100
    colormap = colormap.astype(np.uint8)
    
    print("Colormap shape:", colormap.shape)
    print("Colormap values:", colormap[:NUM_CLASSES])

    plot_predictions(train_images[:4], colormap, model=model)
    
    
    

if __name__ == '__main__':
    main()