import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import pathlib
import json

dataset_path = "C:/Users/adr/.cache/kagglehub/datasets/ma7555/cat-breeds-dataset/versions/2"
dataset_img_folder = dataset_path + "/images"
img_width, img_height = 180, 180
batch_size = 256
validation_split = 0.2

image_builder = tfds.ImageFolder(dataset_img_folder)

data_dir = pathlib.Path(dataset_img_folder).with_suffix('')
image_count = len(list(data_dir.glob('*/*.jpg')))

auto = tf.data.experimental.AUTOTUNE

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.8,
    subset = 'training',
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = 'validation',
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)

### Standaridizing the data
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x),y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

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
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, 3, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, 3, activation = 'relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes)
])


model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True)

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 3,
    callbacks = [early_stopping]
)

model.save('model_of_cats2.keras')
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)