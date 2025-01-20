from TrainMethods import data_generator, DeeplabV3
from InferMethods import plot_predictions

from scipy.io import loadmat
import numpy as np
import os
from glob import glob
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

IMAGE_SIZE = 512
BATCH_SIZE = 4
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
    
    print("Train Dataset: ", train_dataset)
    print("Val Dataset: ", val_dataset)   
    
    
    # [2] 모델 학습. 
    # [2-1] GPU 메모리 설정 (선택 사항)
    #    - TensorFlow가 처음에 모든 GPU 메모리를 할당하지 않고, 필요한 만큼만 점차 할당하도록 설정
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # [2-2] MirroredStrategy 초기화
    #    - 기본적으로 모든 GPU를 자동으로 감지해 분산 학습을 설정해 줍니다.
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = DeeplabV3(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
        model.summary()

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss=loss,
                      metrics=['accuracy'])


    history = model.fit(train_dataset, validation_data=val_dataset, epochs=30)

    # [3] 모델 저장
    # [3-1] HDF5 형식으로 저장
    model.save('my_model.h5')  # .h5 파일 형식으로 저장

    # [3-2] TensorFlow SavedModel 형식으로 저장
    model.save('saved_model/my_model')  # 디렉토리로 저장

    print("모델이 성공적으로 저장되었습니다.")

    # [4] 차트를 활용하여 loss와 accuracy 살펴보기
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig = plt.figure(figsize=(12,5))


    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(loss, color='blue', label='train_loss')
    ax1.plot(val_loss, color='red', label='val_loss')
    ax1.set_title('Train and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid()
    ax1.legend()

    accracy = history.history['accracy']
    val_accracy = history.history['val_accracy']

    ax2 = fig.add_subplot(1, 2, 1)
    ax2.plot(accracy, color='blue', label='train_accuracy')
    ax2.plot(val_accracy, color='red', label='val_accracy')
    ax2.set_title('Train and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid()
    ax2.legend()

    plt.show()
    
    
    # [5] 추론 
    # 데이터셋과 함께 제공된 ./instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat 파일을 통해 각 라벨에 대한 해당 색상을 찾을 수 있음
    colormap = loadmat('./instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat')['colormap']
    colormap = colormap * 100
    colormap = colormap.astype(np.uint8)

    plot_predictions(train_images[:4], colormap, model=model)
    

if __name__ == '__main__':
    main()