from datasets import load_dataset
import os

# Define the save path
save_path = "output2"  # Change this to your desired path

# Load and save the dataset
# dataset_small = load_dataset("pierreguillou/DocLayNet-small", cache_dir=save_path)
dataset_small = load_dataset("ds4sd/DocLayNet", split="val", cache_dir=save_path)

# Get the total size of the downloaded dataset
def get_directory_size(directory):
    total_size = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# Compute and print the size in MB
dataset_size = get_directory_size(save_path) / (1024 * 1024)
print(f"Dataset downloaded to: {save_path}")
print(f"Total dataset size: {dataset_size:.2f} MB")
