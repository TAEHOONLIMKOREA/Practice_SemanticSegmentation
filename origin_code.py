import os
import cv2
import numpy as np
from glob import glob
from scipy.io import loadmat
import json

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

IMAGE_SIZE = 512
BATCH_SIZE = 16
NUM_CLASSES = 20
DATA_DIR = "./instance-level_human_parsing/instance-level_human_parsing/Training"
NUM_TRAIN_IMAGES = 1000
NUM_VAL_IMAGES = 50

# 1. 경로 수집
train_images = sorted(glob(os.path.join(DATA_DIR, 'Images/*')))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, 'Category_ids/*')))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, 'Images/*')))[NUM_TRAIN_IMAGES:NUM_TRAIN_IMAGES+NUM_VAL_IMAGES]
val_masks = sorted(glob(os.path.join(DATA_DIR, 'Category_ids/*')))[NUM_TRAIN_IMAGES:NUM_TRAIN_IMAGES+NUM_VAL_IMAGES]

# 2. 데이터 로드 함수
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
    image = read_image(image_list, mask=False)
    mask = read_image(mask_list, mask=True)
    return image, mask

def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

train_dataset = data_generator(train_images, train_masks)
val_dataset   = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)
print("Val Dataset:",   val_dataset)

# 3. 모델 구현부
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
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2]),
        interpolation='bilinear'
    )(x)
    
    # Dilated Convs
    out_1  = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6  = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

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

# 4. 디버깅용 - 검증 마스크 픽셀 분포 확인
test_mask = read_image(val_masks[0], mask=True).numpy()
unique_vals, counts = np.unique(test_mask, return_counts=True)
print("Mask unique values:", unique_vals)
print("Counts:", counts)
print("Mask shape:", test_mask.shape, "dtype:", test_mask.dtype)

print(len(val_images), len(val_masks))

# 5. 모델 학습
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = DeeplabV3(num_classes=NUM_CLASSES)
    model.summary()

    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    optimizer = Adam(learning_rate=0.0001)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30
    )

# 6. 학습 결과 시각화
loss = history.history['loss']
val_loss = history.history['val_loss']

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(loss, label='train_loss', color='blue')
ax1.plot(val_loss, label='val_loss', color='red')
ax1.set_title('Train and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid()
ax1.legend()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(accuracy, label='train_accuracy', color='blue')
ax2.plot(val_accuracy, label='val_accuracy', color='red')
ax2.set_title('Train and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.grid()
ax2.legend()

plt.savefig("acc_loss_origin")

# 7. 추론 관련 함수
colormap = loadmat('./instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat')['colormap']
colormap = (colormap * 100).astype(np.uint8)

def infer(model, image_tensor):
    predictions = model.predict(tf.expand_dims(image_tensor, axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask, dtype=np.uint8)
    g = np.zeros_like(mask, dtype=np.uint8)
    b = np.zeros_like(mask, dtype=np.uint8)
    for i in range(n_classes):
        idx = mask == i
        r[idx] = colormap[i, 0]
        g[idx] = colormap[i, 1]
        b[idx] = colormap[i, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def get_overlay(image, colored_mask):
    # image_tensor → uint8 변환
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    # 겹쳐보기
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay

def plot_samples_matplotlib(display_list, figsize=(5,3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.savefig("result")

def plot_predictions(images_list, colormap, model):
    for image_file in images_list:
        image_tensor = read_image(image_file, mask=False)
        prediction_mask = infer(model, image_tensor)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, NUM_CLASSES)
        overlay = get_overlay(image_tensor, prediction_colormap)
        plot_samples_matplotlib([image_tensor, overlay, prediction_colormap], figsize=(18, 14))

# 8. 예시 추론
plot_predictions(train_images[:4], colormap, model=model)

# 9. 모델 저장
model.save('my_model.h5')                # HDF5 포맷
model.save('saved_model/my_model')       # SavedModel 포맷
print("모델이 성공적으로 저장되었습니다.")

# 10. history 저장
with open('history.json', 'w') as f:
    json.dump(history.history, f)
