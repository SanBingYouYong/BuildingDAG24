import yaml
import os
from PIL import Image
from tqdm import tqdm
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from nn_models import *
from nn_dataset import *
from nn_training import *


if __name__ == "__main__":
    '''
    Different splitting yields different results (test score more similar to train)
    '''
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)

    dataset_name = "DAGDataset100_100_5"
    if not os.path.exists(f"./datasets/{dataset_name}"):
        raise FileNotFoundError(f"Dataset {dataset_name} not found")

    decoder = "Building Mass Decoder"
    model_name = "model_DAGDataset100_100_5_20240325102238"
    weights_path = f"./models/{model_name}.pth"
    # Load metadata
    # ranges, parameter_output_mapping, decoders, switches, batch_cam_angles = load_metadata_for_inference(f"./models/{model_name}_meta.yml", need_full=True, decoder=decoder)
    ranges, parameter_output_mapping, decoders, switches, batch_cam_angles = load_metadata_for_inference(f"./models/{model_name}_meta.yml", need_full=True)

    # Load the dataset
    # dataset = DAGDatasetSingleDecoder(decoder, dataset_name)
    dataset = DAGDataset(dataset_name)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.8, 0.1, 0.1)
    train_loader, val_loader, test_loader = create_dataloaders_of(train_dataset, val_dataset, test_dataset, batch_size=128)

    # Load the model
    # encoder = Encoder()
    # model = EncoderDecoderModel(encoder, decoders)
    model = ManualEncoderDecoderModelBM()
    model.load_state_dict(torch.load(f"./models/{model_name}.pth", map_location=device
    ))
    model.eval()
    model.to(device)

    # Loss function
    # criterion = EncDecsLoss(decoders, switches)
    criterion = custom_loss

    # Inference
    test(model, weights_path, test_loader, criterion, ranges, results_save_path="results.yml")
