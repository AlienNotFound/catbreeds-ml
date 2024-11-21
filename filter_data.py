import os
import cv2

dataset_dir = "C:/Users/adr/.cache/kagglehub/datasets/denispotapov/cat-breeds-dataset-cleared/versions/2/dataset/images/Pixiebob"

def filter_corrupt_images(dataset_dir):
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg")):
                file_path = os.path.join(root, file)

                print(file_path)
                cv2.imread(file_path)

filter_corrupt_images(dataset_dir)