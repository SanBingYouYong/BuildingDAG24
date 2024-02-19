import yaml
import os
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from nn_models import *
from nn_dataset import *


def load_ranges(dataset_name: str, datasets_folder: str="./datasets"):
    dataset_path = os.path.join(datasets_folder, dataset_name)
    ranges_file_path = os.path.join(dataset_path, "ranges.yml")
    with open(ranges_file_path, 'r') as file:
        ranges = yaml.safe_load(file)
    # Create a mapping between parameter names and output sizes
    parameter_output_mapping = {}
    for decoder_name, param_specs in ranges.items():
        if param_specs['type'] == 'float':
            parameter_output_mapping[decoder_name] = 1  # 1 for scalar
        elif param_specs['type'] == 'int':
            parameter_output_mapping[decoder_name] = 1  # 1 for scalar
        elif param_specs['type'] == 'vector':
            parameter_output_mapping[decoder_name] = 3  # 3 for x, y, z
        elif param_specs['type'] == 'states':
            parameter_output_mapping[decoder_name] = len(param_specs['values'])
        elif param_specs['type'] == 'bool':
            parameter_output_mapping[decoder_name] = 2  # 2 for binary encoding
    return ranges, parameter_output_mapping

def load_decoders(dataset_name: str, ranges: dict, parameter_output_mapping: dict, datasets_folder: str="./datasets"):
    dataset_path = os.path.join(datasets_folder, dataset_name)
    decoders_file_path = os.path.join(dataset_path, "decoders.yml")
    with open(decoders_file_path, 'r') as file:
        decoders_params = yaml.safe_load(file)
    decoders = nn.ModuleDict()
    # initialize decoders with corresponding output tails
    for decoder_name, param_names in decoders_params.items():
        classification_tails = {}
        regression_tails = {}
        for param_name in param_names:
            spec = ranges[param_name]
            # if type is bool or states, add to classification tails
            # if type is float, int or vector, add to regression tails
            if spec['type'] == 'bool' or spec['type'] == 'states':
                classification_tails[param_name] = parameter_output_mapping[param_name]
            else:
                regression_tails[param_name] = parameter_output_mapping[param_name]
        # add decoder to model
        decoders[decoder_name] = ParamAwareMultiTailDecoder(1024, classification_tails, regression_tails)
    return decoders

def load_switches(dataset_name: str, datasets_folder: str="./datasets"):
    dataset_path = os.path.join(datasets_folder, dataset_name)
    switches_file_path = os.path.join(dataset_path, "switches.yml")
    with open(switches_file_path, 'r') as file:
        switches = yaml.safe_load(file)
    return switches

def train(model: nn.Module, criterion: nn.Module, optimizer, train_loader, val_loader, epochs=25, seed=0, model_save_path="EncDecModel.pth"):
    train_losses = []
    val_losses = []

    torch.manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch + 1}", leave=False)

        for i, data in progress_bar:
            inputs, targets = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"training_loss": train_loss / (i + 1)})

        train_losses.append(train_loss / num_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        num_batches_val = len(val_loader)

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, targets = data

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_losses.append(val_loss / num_batches_val)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}")

    print("Finished Training")

    # Save your trained model if needed
    torch.save(model.state_dict(), model_save_path)

def test(model: nn.Module, test_loader, criterion: nn.Module):
    model.eval()
    test_loss = 0.0
    num_batches = len(test_loader)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, targets = data

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / num_batches}")

    return outputs, targets


if __name__ == "__main__":
    dataset_name = "DAGDataset100_100_5"
    dataset = DAGDataset(dataset_name)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.8, 0.1, 0.1)
    train_loader, val_loader, test_loader = create_dataloaders_of(train_dataset, val_dataset, test_dataset, batch_size=32)
    encoder = Encoder()
    ranges, parameter_output_mapping = load_ranges(dataset_name)
    decoders = load_decoders(dataset_name, ranges, parameter_output_mapping)
    model = EncoderDecoderModel(encoder, decoders)
    switches = load_switches(dataset_name)
    criterion = EncDecsLoss(decoders, switches)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train(model, criterion, optimizer, train_loader, val_loader, epochs=25, seed=0, model_save_path="EncDecModel.pth")
    outputs, targets = test(model, test_loader, criterion)
