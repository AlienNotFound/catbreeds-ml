import os
from os import listdir
import cv2
from PIL import Image
import logging

dataset_dir = "C:/Users/adr/.cache/kagglehub/datasets/ma7555/cat-breeds-dataset/versions/2/imgs_to_test"
corrupt_dir = "C:/Users/adr/.cache/kagglehub/datasets/ma7555/cat-breeds-dataset/versions/2/corrupt_images"
os.makedirs(corrupt_dir, exist_ok = True)

logging.basicConfig(filename = "filtered_corrupt_images.log", level = logging.INFO)

def filter_corrupt_images(dataset_dir, corrupt_dir):
    valid_image_paths = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg")):
                file_path = os.path.join(root, file)

                print(file_path)
                cv2.imread(file_path)

                # try:
                #     with Image.open(file_path) as img:
                #         # img.verify()
                #         img.load()
                #         # print(img)
                #         # print(file_path)
                #     # valid_image_paths.append(file_path)
                # except (IOError, SyntaxError):
                #     logging.info(f"Corrupt image: {file_path}")
                #     corrupt_path = os.path.join(corrupt_dir, file)
                #     os.replace(file_path, corrupt_path)
    
    return valid_image_paths

valid_image_paths = filter_corrupt_images(dataset_dir, corrupt_dir)

# with open("valid_image_paths.txt", "w") as f:
#     for path in valid_image_paths:
#         f.write(path + "\n")