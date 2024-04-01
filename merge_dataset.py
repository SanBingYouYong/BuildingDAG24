import yaml
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from tqdm import tqdm
import shutil

def merge_datasets(merged_dataset_name: str, datasets_to_merge: list, datasets_folder: str):
    for dataset in datasets_to_merge:
        if not os.path.exists(f"{datasets_folder}/{dataset}"):
            raise FileNotFoundError(f"Dataset {dataset} not found")
    # prep the new dataset directory
    os.makedirs(f"{datasets_folder}/{merged_dataset_name}", exist_ok=True)
    os.makedirs(f"{datasets_folder}/{merged_dataset_name}/images", exist_ok=True)
    os.makedirs(f"{datasets_folder}/{merged_dataset_name}/params", exist_ok=True)
    # merge
    batch_count = 0
    new_meta = {}
    # assumption: data ranges are the same, thus only need to merge batch_cam_angles and acccumulate new batch numbers
    for i, dataset in enumerate(datasets_to_merge):
        print(f"Merging {dataset}...")
        # read meta
        with open(f"{datasets_folder}/{dataset}/meta.yml", "r") as f:
            meta = yaml.safe_load(f)
            batch_cam_angles:dict = meta["batch_cam_angles"]
        if new_meta == {}:
            new_meta = meta
            new_meta["batch_cam_angles"] = {}
        accum_batch_cam_angles = {}
        # read images and params
        images = os.listdir(f"{datasets_folder}/{dataset}/images")
        params = os.listdir(f"{datasets_folder}/{dataset}/params")
        # find total number of batches
        total_batches = len(batch_cam_angles)
        # wrap the loop with tqdm for progress bar
        with tqdm(total=total_batches, desc=f'Merging {dataset} images') as pbar:
            # rename current batches
            for batch, cam_angle in batch_cam_angles.items():
                # get batch number from "batch<num>"
                batch_num = int(batch.split("batch")[1])
                accum_batch_num = batch_num + batch_count
                accum_batch_cam_angles[f"batch{accum_batch_num}"] = cam_angle
                # copy images
                for image in images:
                    if image.startswith(f"batch{batch_num}"):
                        shutil.copy(f"{datasets_folder}/{dataset}/images/{image}", f"{datasets_folder}/{merged_dataset_name}/images/batch{accum_batch_num}_{image.split('_')[1]}")
                        pbar.update(1)  # update progress bar
                # copy params
                for param in params:
                    if param.startswith(f"batch{batch_num}"):
                        shutil.copy(f"{datasets_folder}/{dataset}/params/{param}", f"{datasets_folder}/{merged_dataset_name}/params/batch{accum_batch_num}_{param.split('_')[1]}")
                pbar.update(1)  # update progress bar for each batch
        batch_count += total_batches
        new_meta["batch_cam_angles"].update(accum_batch_cam_angles)
    # write new meta
    new_meta["dataset"] = merged_dataset_name
    with open(f"{datasets_folder}/{merged_dataset_name}/meta.yml", "w") as f:
        yaml.dump(new_meta, f)



if __name__ == "__main__":
        
    datasets_to_merge = ["DAGDataset100_100_5", "DAGDataset100_100_5_0", "DAGDataset300_100_5"]
    datasets_folder = "./datasets"

    merge_datasets("DAGDataset500_100_5", datasets_to_merge, datasets_folder)
    print("Datasets merged successfully.")
