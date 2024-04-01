import yaml
import os

path = "./datasets/DAGDataset500_100_5/params"

# count images
count = 0
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".yml"):
            count += 1
print(count)
