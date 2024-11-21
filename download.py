import kagglehub

# Download latest version
path = kagglehub.dataset_download("denispotapov/cat-breeds-dataset-cleared")

print("Path to dataset files:", path)