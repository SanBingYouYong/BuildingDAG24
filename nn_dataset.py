import yaml
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class DAGDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, datasets_folder: str="./datasets", transform=None, device=None):
        self.dataset_name = dataset_name
        self.datasets_folder = datasets_folder
        self.dataset_path = os.path.join(self.datasets_folder, self.dataset_name)
        self.images_folder = os.path.join(self.dataset_path, "images")
        self.params_folder = os.path.join(self.dataset_path, "params")
        self.ranges_file_path = os.path.join(self.dataset_path, "ranges.yml")
        self.ranges = None
        self.decoders = None
        self.transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ) if transform is None else transform
        self.data = self.load_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    def load_data(self):
        # Load the ranges from the YAML file
        with open(self.ranges_file_path, 'r') as file:
            self.ranges = yaml.safe_load(file)
        # Load the decoders from the YAML file
        with open(os.path.join(self.dataset_path, "decoders.yml"), 'r') as file:
            self.decoders = yaml.safe_load(file)
        # read images and parameters
        data = []
        for image_name in os.listdir(self.images_folder):
            image_path = os.path.join(self.images_folder, image_name)
            param_path = os.path.join(self.params_folder, os.path.splitext(image_name)[0] + ".yml")
            with open(param_path, 'r') as file:
                param = yaml.safe_load(file)
            # normalize
            param = self.preprocess(param)
            param = self.format_target_to_decoders(param)
            data.append((image_path, param))
        return data

    def format_target_to_decoders(self, target):
        formatted_target = {}
        for decoder_name, decoder_params in self.decoders.items():
            formatted_target[decoder_name] = {
                "classification_targets": {},
                "regression_target": {}
            }
            for param_name in decoder_params:
                param_type = self.ranges[param_name]['type']
                if param_type == 'float' or param_type == 'int' or param_type == 'vector':
                    formatted_target[decoder_name]['regression_target'][param_name] = target[param_name]
                elif param_type == 'states' or param_type == 'bool':
                    formatted_target[decoder_name]['classification_targets'][param_name] = target[param_name]
        return formatted_target

    def preprocess(self, param):
        processed_param = {}
        # for float and vector: normalize with min max
        # for states, bool: convert to one hot
        # for ints: treat as float, but round back to int when saving as param
        for param_name, param_spec in self.ranges.items():
            if param_spec['type'] == 'float' or param_spec['type'] == 'int' or param_spec['type'] == 'vector':
                processed_param[param_name] = self.normalize(param[param_name], param_spec)
            elif param_spec['type'] == 'states' or param_spec['type'] == 'bool':
                processed_param[param_name] = self.one_hot(param[param_name], param_spec)
            else:
                raise ValueError(f"Unsupported parameter type: {param_spec['type']}")
        return processed_param

    def normalize(self, value, param_spec):
        if param_spec['type'] == 'float' or param_spec['type'] == 'int':
            return (value - param_spec['min']) / (param_spec['max'] - param_spec['min'])
        elif param_spec['type'] == 'vector':
            return [(value[i] - param_spec[f'{dim}min']) / (param_spec[f'{dim}max'] - param_spec[f'{dim}min']) for i, dim in enumerate(['x', 'y', 'z'])]
        else:
            raise ValueError(f"Unsupported parameter type: {param_spec['type']}")

    def one_hot(self, value, param_spec):
        if param_spec['type'] == 'states':
            index = param_spec['values'].index(value)
            return [1 if i == index else 0 for i in range(len(param_spec['values']))]
        elif param_spec['type'] == 'bool':
            # make bools onehot too to make it consistent
            return [1, 0] if value else [0, 1]
        else:
            raise ValueError(f"Unsupported parameter type: {param_spec['type']}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, target = self.data[idx]
        sample = Image.open(sample).convert('L')

        if self.transform:
            sample = self.transform(sample)

        # convert target's values to tensor
        for decoder_name, decoder_outputs in target.items():
            # check if is tensor
            for classification_target in decoder_outputs['classification_targets']:
                if not torch.is_tensor(decoder_outputs['classification_targets'][classification_target]):
                    target[decoder_name]['classification_targets'][classification_target] = torch.tensor(decoder_outputs['classification_targets'][classification_target], dtype=torch.float32)
            for regression_target in decoder_outputs['regression_target']:
                if not torch.is_tensor(decoder_outputs['regression_target'][regression_target]):
                    target[decoder_name]['regression_target'][regression_target] = torch.tensor(decoder_outputs['regression_target'][regression_target], dtype=torch.float32)

        # move to device
        sample = sample.to(self.device)
        target = {decoder_name: {task: {param_name: param.to(self.device) for param_name, param in task_params.items()} for task, task_params in decoder_outputs.items()} for decoder_name, decoder_outputs in target.items()}

        return sample, target


def split_dataset(dataset: DAGDataset, train_ratio: float=0.8, val_ratio: float=0.1, test_ratio: float=0.1):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

def create_dataloaders_of(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader

def overfit_dataloaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader
