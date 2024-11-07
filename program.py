import PIL.Image
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
import PIL
import numpy as np

dataset_path = "C:/Users/adr/.cache/kagglehub/datasets/ma7555/cat-breeds-dataset/versions/2"
dataset_csv = dataset_path + "/data/cats.csv"
dataset_img_folder = dataset_path + "/images"

## LINKS
# https://www.tensorflow.org/tutorials/load_data/images

image_builder = tfds.ImageFolder(dataset_img_folder)

data_dir = pathlib.Path(dataset_img_folder).with_suffix('')
image_count = len(list(data_dir.glob('*/*.jpg')))

aby = list(data_dir.glob('Abyssinian/*'))

auto = tf.data.experimental.AUTOTUNE
batch_size = 256

img_width = 180
img_height = 180

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

print(class_names)
print("Number of classes: " + str(len(class_names)))

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
num_classes = len(class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(input_shape = (180, 180)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)

model.__setattr__("class_names", class_names)

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 3
)

model.save('model_of_cats.keras')


# PIL.Image.open(str(aby[0]))
print(image_count)




### CVS file
# df = pd.read_csv(dataset_csv)
# df = df[['url', 'breed', 'photos', 'med_photos']]

# df = df.head()

# print(image_builder.info)