import yaml
import os

path = "./datasets/DAGDataset10_10_5/images"

# count images
count = 0
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".png"):
            count += 1
print(count)
