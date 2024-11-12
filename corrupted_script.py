# def load_image_with_logging(image_path):
#     try:
#         img = tf.io.read_file(image_path)
#         img = tf.image.decode_jpeg(img, channels = 3)
#         img = tf.image.resize(img, [img_height, img_width])
#         img = img / 255.0
#         return img, True
#     except (tf.errors.InvalidArgumentError, tf.errors.NotFoundError) as e:
#         logging.info(f"Corrupt image: {image_path}: {e}")
#         return None, False
    
# def create_dataset_with_logging(data_dir, batch_size, validation_split):
#     image_paths = [os.path.join(root, fname)
#                    for root, _, files in os.walk(data_dir)
#                    for fname in files if fname.endswith((".jpg", ".jpeg"))]

#     split_index = int(len(image_paths) * (1 - validation_split))
#     train_paths = image_paths[:split_index]
#     val_paths = image_paths[split_index:]

#     def filter_valid_images(paths):
#         valid_paths = []
#         for path in paths:
#             _, is_valid = load_image_with_logging(path)
#             if is_valid:
#                 valid_paths.append(path)
#         return valid_paths
    
#     train_paths = filter_valid_images(train_paths)
#     val_paths = filter_valid_images(val_paths)

#     def process_path(file_path):
#         img, _ = load_image_with_logging(file_path)
#         label = tf.strings.split(file_path, os.sep)[-2]
#         return img, label
    
#     train_ds = tf.data.Dataset.from_tensor_slices(train_paths)
#     train_ds = train_ds.map(lambda x: process_path(x), num_parallel_calls = tf.data.AUTOTUNE)
#     train_ds = train_ds.batch(batch_size).prefetch(buffer_size = tf.data.AUTOTUNE)

#     val_ds = tf.data.Dataset.from_tensor_slices(val_paths)
#     val_ds = val_ds.map(lambda x: process_path(x), num_parallel_calls = tf.data.AUTOTUNE)
#     val_ds = val_ds.batch(batch_size).prefetch(buffer_size = tf.data.AUTOTUNE)

#     return train_ds, val_ds, image_paths

    # valid_image_paths = []
    # for image_path in image_paths:
    #     img, is_valid = load_image_with_logging(image_path, img_height, img_width)
    #     if is_valid:
    #         valid_image_paths.append(image_path)

    # dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    # dataset = dataset.map(lambda x: load_image_with_logging(x, img_height, img_width))
    # dataset = dataset.batch(batch_size)

    # return dataset

# train_ds, val_ds, image_paths = create_dataset_with_logging(dataset_img_folder, batch_size, validation_split)
#class_names = list(set([pathlib.Path(p).parent.name for p in image_paths]))

#data_dir = dataset_img_folder
# train_ds = create_dataset_with_logging(data_dir, img_height = 180, img_width = 180)

    # tf.keras.layers.Conv2D(32, 3, activation = 'relu', input_shape = (img_height, img_width, 3)),
