import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat

import tensorflow as tf
from tensorflow import keras
from keras import layers, initializers
import matplotlib.pyplot as plt

import tensorflow as tf

IMAGE_SIZE = 512
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

def data_generator(image_list, mask_list, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def convolution_block(block_input, num_filters=256, kernel_size=3,
                      dilation_rate=1, padding='same', use_bias=False):
    x = layers.Conv2D(num_filters, kernel_size, dilation_rate=dilation_rate,
                      padding='same', use_bias=use_bias, kernel_initializer=initializers.HeNormal())(block_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2]), interpolation='bilinear')(x)
    
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
    model_input = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    resnet50 = keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=model_input)
    
    x = resnet50.get_layer('conv4_block6_2_relu').output
    x = DilatedSpatialPyramidPooling(x)    
    input_a = layers.UpSampling2D(size=(IMAGE_SIZE // 4 // x.shape[1],
                                         IMAGE_SIZE // 4 // x.shape[2]),
                                   interpolation='bilinear')(x)
    
    input_b = resnet50.get_layer('conv2_block3_2_relu').output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)
    
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(size=(IMAGE_SIZE // x.shape[1],
                                  IMAGE_SIZE // x.shape[2]),
                            interpolation='bilinear')(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1,1), padding='same')(x)
    return keras.Model(inputs=model_input, outputs=model_output)

