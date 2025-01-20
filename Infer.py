# # ----------- 추론 -----------
# # 데이터셋과 함께 제공된 ./instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat 파일을 통해 각 라벨에 대한 해당 색상을 찾을 수 있음
# colormap = loadmat('./instance-level_human_parsing/instance-level_human_parsing/human_colormap.mat')['colormap']
# colormap = colormap * 100
# colormap = colormap.astype(np.uint8)

# def infer(model, image_tensor):
#     predictions = model.predict(np.expand_dims((image_tensor), axis=0))
#     predictions = np.squeeze(predictions)
#     predictions = np.argmax(predictions, axis=2)
#     return predictions

# def decode_segmentation_masks(mask, colormap, n_classes):
#     r = np.zeros_like(mask).astype(np.uint8)
#     g = np.zeros_like(mask).astype(np.uint8)
#     b = np.zeros_like(mask).astype(np.uint8)
#     for i in range(0, n_classes):
#         idx = mask == i
#         r[idx] = colormap[i, 0]
#         g[idx] = colormap[i, 1]
#         b[idx] = colormap[i, 2]
#     rgb = np.stack([r,g,b], axis=2)
#     return rgb

# def get_overlay(image, colored_mask):
#     image = tf.keras.preprocessing.image.array_to_img(image)
#     image = np.array(image).astype(np.uint8)
#     overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
#     return overlay


# def plot_samples_matplotlib(display_list, figsize=(5,3)):
#     _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
#     for i in range(len(display_list)):
#         if display_list[i].shape[-1] == 3:
#             axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
#         else:
#             axes[i].imshow(display_list[i])
#     plt.show()
    
# def plot_predictions(images_list, colormap, model):
#     for image_file in images_list:
#         image_tensor = read_image(image_file)
#         prediction_mask = infer(image_tensor = image_tensor, model=model)
#         prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
#         overlay = get_overlay(image_tensor, prediction_colormap)
#         plot_samples_matplotlib([image_tensor, overlay, prediction_colormap], figsize=(18, 14))
        

# plot_predictions(train_images[:4], colormap, model=model)