from glob import glob
from scipy.io import loadmat
import json
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


IMAGE_SIZE = 512
BATCH_SIZE = 16
NUM_CLASSES = 20


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        # 세그멘테이션 마스크는 Nearest Neighbor로 리사이즈 권장(범주형 레이블 보존)
        image = tf.image.resize(
            images=image, 
            size=[IMAGE_SIZE, IMAGE_SIZE],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        # 필요하면 정수형 캐스팅
        image = tf.cast(image, tf.uint8)
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])  # RGB
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        # -1 ~ 1 범위로 정규화
        image = image / 127.5 - 1.0
    return image

def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask

def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


def convolution_block(block_input, num_filters=256, kernel_size=3,
                      dilation_rate=1, padding='same', use_bias=False):
    x = layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=HeNormal()
    )(block_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    # Global Average Pooling branch
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2]), interpolation='bilinear')(x)
    
    # Dilated Convs
    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)
    
    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


# ----------- 모델 구성 -----------
# 사전 훈련된 ResNet50을 백본 모델로 사용
# conv4_block6_2_relu 블록에서 저수준의 특징 사용
# 인코더 특징은 인자 4에 의해 쌍선형 업샘플링
# 동일한 공간 해상도를 가진 네트워크 백본에서 저수준 특징과 연결

def DeeplabV3(num_classes):
    # 입력 정의
    model_input = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    # ResNet50 백본
    resnet50 = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=model_input
    )

    # conv4_block6_2_relu 출력
    x = resnet50.get_layer('conv4_block6_2_relu').output
    x = DilatedSpatialPyramidPooling(x)

    # Upsampling 1
    input_a = layers.UpSampling2D(
        size=(IMAGE_SIZE // 4 // x.shape[1], IMAGE_SIZE // 4 // x.shape[2]),
        interpolation='bilinear'
    )(x)

    # 저수준 특징: conv2_block3_2_relu
    input_b = resnet50.get_layer('conv2_block3_2_relu').output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    # Feature Fusion
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)

    # 최종 업샘플
    x = layers.UpSampling2D(
        size=(IMAGE_SIZE // x.shape[1], IMAGE_SIZE // x.shape[2]),
        interpolation='bilinear'
    )(x)
    # 클래스 채널
    model_output = layers.Conv2D(num_classes, kernel_size=(1,1), padding='same')(x)

    return tf.keras.Model(inputs=model_input, outputs=model_output)

def train(train_dataset, val_dataset):
    # [1] 모델 학습. 
    # [1-1] GPU 메모리 설정 (선택 사항)
    #    - TensorFlow가 처음에 모든 GPU 메모리를 할당하지 않고, 필요한 만큼만 점차 할당하도록 설정
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # [1-2] MirroredStrategy 초기화
    #    - 기본적으로 모든 GPU를 자동으로 감지해 분산 학습을 설정해 줍니다.
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = DeeplabV3(num_classes=NUM_CLASSES)
        model.summary()
        print("Model output shape:", model.output_shape)

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                      loss=loss,
                      metrics=['accuracy'])
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=30)

    # [2] 모델 저장
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # [2-1] HDF5 형식으로 저장
    model.save('model_' + formatted_time + '.h5')  # .h5 파일 형식으로 저장

    # [2-2] TensorFlow SavedModel 형식으로 저장
    model.save('saved_model/model_'+ formatted_time)  # 디렉토리로 저장

    print("모델이 성공적으로 저장되었습니다.")

    # [3] history 저장
    with open('history.json', 'w') as f:
        json.dump(history.history, f)
    # [3-1] 차트를 활용하여 loss와 accuracy 살펴보기
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

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(accuracy, color='blue', label='train_accuracy')
    ax2.plot(val_accuracy, color='red', label='val_accuracy')
    ax2.set_title('Train and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid()
    ax2.legend()
    
    plt.savefig("Acc_Loss_" + formatted_time)
    
    return model