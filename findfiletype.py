
import os, os.path
import filetype

folder_path = 'C:/Users/adr/.cache/kagglehub/datasets/ma7555/cat-breeds-dataset/versions/2/images/Abyssinian'
count = 0

for path in os.listdir(folder_path):
    if os.path.isfile(os.path.join(folder_path, path)):
        count += 1
        print(filetype.guess(path).extension)
