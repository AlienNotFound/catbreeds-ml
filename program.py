import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pathlib
import json

dataset_path = "C:/Users/adr/.cache/kagglehub/datasets/denispotapov/cat-breeds-dataset-cleared/versions/2/dataset"
dataset_img_folder = dataset_path + "/images"
img_width, img_height = 180, 180
batch_size = 32
validation_split = 0.2

image_builder = tfds.ImageFolder(dataset_img_folder)

data_dir = pathlib.Path(dataset_img_folder).with_suffix('')
image_count = len(list(data_dir.glob('*/*.jpg')))

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = validation_split,
    subset = 'training',
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = validation_split,
    subset = 'validation',
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)

### Standaridizing the data
AUTOTUNE = tf.data.AUTOTUNE

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x),y), num_parallel_calls = AUTOTUNE)
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

train_ds = train_ds.prefetch(buffer_size = AUTOTUNE)#.cache()
val_ds = val_ds.prefetch(buffer_size = AUTOTUNE)#.cache()

### Model training

## Data augmentation
#  Creating variations of the data, while training to get a better understanding of the material.
#  Such as rotating, brightness adjustments and flipping the image.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'), # Filter window. 32 = number of filters, 3 = height and width of filter
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(), # Takes a small window (default 2x2), saves the most prominent feature in that window of the picture and moves on to the next window
    tf.keras.layers.Dropout(0.2), # Forced the model to use different paths and neurons, so it doesn't "get used" to one path
    tf.keras.layers.Conv2D(64, 3, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, 3, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes)
])

# model = tf.keras.models.load_model('model_of_cats.keras')

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 3,
    restore_best_weights = True,
    start_from_epoch = 2,
    verbose = 1)


model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 10,
    callbacks = [early_stopping]
)

model.save('model_of_cats_short.keras')
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

# print("Finished training.") # Had issues with the model ending after the first epoch run, without feedback. Printing this for verification that it reaches end of code.