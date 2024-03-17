import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


import os
import yaml
from PIL import Image
from tqdm import tqdm

from nn_manual_single_test import SingleEncoderDecoderModel, SingleTaskDataset, test

# load weights to model
model = SingleEncoderDecoderModel().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.load_state_dict(torch.load('./models/single_encdec.pth'))
model.eval()

dataset = SingleTaskDataset(task_param_name='Bm Base Shape', dataset_name='DAGDataset100_100_5')

# Split the dataset into training, validation, and test subsets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = test_size = (len(dataset) - train_size) // 2  # Remaining 10% each for validation and test

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders for training, validation, and test
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test(model, test_dataloader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
