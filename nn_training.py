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

# from nn_models import *
from nn_overfit_model import *
from nn_dataset import *


def load_ranges(metadata: dict):
    ranges = metadata['ranges']
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

def load_decoders(metadata: dict, ranges: dict, parameter_output_mapping: dict, decoder_input_size=1024):
    decoders_params = metadata['decoders']
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
        decoders[decoder_name] = ParamAwareMultiTailDecoder(decoder_input_size, classification_tails, regression_tails)
    return decoders

def load_metadata(dataset_name: str, datasets_folder: str="./datasets"):
    dataset_path = os.path.join(datasets_folder, dataset_name)
    metadata_file_path = os.path.join(dataset_path, "meta.yml")
    with open(metadata_file_path, 'r') as file:
        metadata = yaml.safe_load(file)
    ranges, parameter_output_mapping = load_ranges(metadata)
    decoders = load_decoders(metadata, ranges, parameter_output_mapping)
    switches = metadata['switches']
    batch_cam_angles = metadata['batch_cam_angles']
    return ranges, parameter_output_mapping, decoders, switches, batch_cam_angles

def load_metadata_for_inference(metadata_file_path: str):
    with open(metadata_file_path, 'r') as file:
        metadata = yaml.safe_load(file)
    ranges, parameter_output_mapping = load_ranges(metadata)
    decoders = load_decoders(metadata, ranges, parameter_output_mapping)
    return ranges, parameter_output_mapping, decoders
    

def train(model: nn.Module, criterion: nn.Module, optimizer, train_loader, val_loader, 
          epochs=25, seed=0, 
          model_save_path="./models/EncDecModel.pth", 
          loss_save_path="./models/loss.yml"):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Initialize with a very high value

    torch.manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch + 1}", leave=False)

        for i, data in progress_bar:
            inputs, targets = data

            # print(targets)
            # raise

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

        val_loss /= num_batches_val
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)

    print("Finished Training")
    print(f"Best Validation Loss: {best_val_loss}; at epoch {val_losses.index(best_val_loss) + 1}")

    # Save the loss
    with open(loss_save_path, "w") as f:
        yaml.dump({"train_losses": train_losses, "val_losses": val_losses}, f)


def test(model: nn.Module, test_loader, criterion: nn.Module, results_save_path="results.yml"):
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

    # # Convert tensors to numpy arrays  # TODO: use a similar explicit conversion like in loss calculation
    # outputs = [tensor.numpy().tolist() for tensor in outputs]
    # targets = [tensor.numpy().tolist() for tensor in targets]

    # # save the results
    # with open(results_save_path, "w") as f:
    #     yaml.dump({"outputs": outputs, "targets": targets}, f)

    return outputs, targets


if __name__ == "__main__":
    dataset_name = "DAGDataset2_1_5"
    if not os.path.exists(f"./datasets/{dataset_name}"):
        raise FileNotFoundError(f"Dataset {dataset_name} not found")
    
    ranges, parameter_output_mapping, decoders, switches, batch_cam_angles = load_metadata(dataset_name)

    dataset = DAGDataset(dataset_name)

    # train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.8, 0.1, 0.1)
    # train_loader, val_loader, test_loader = create_dataloaders_of(train_dataset, val_dataset, test_dataset, batch_size=32)
    
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.5, 0.5, 0)
    train_loader, val_loader = overfit_dataloaders(train_dataset, val_dataset, batch_size=32)
    
    print(f"Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    encoder = Encoder()
    model = EncoderDecoderModel(encoder, decoders)

    criterion = EncDecsLoss(decoders, switches)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    os.makedirs("./models", exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model_name = f"./models/model_{dataset_name}_{timestamp}.pth"
    loss_name = f"./models/model_{dataset_name}_{timestamp}_loss.yml"
    train(model, criterion, optimizer, train_loader, val_loader, epochs=100, seed=0, model_save_path=model_name, loss_save_path=loss_name)
    # test(model, test_loader, criterion, results_save_path="results.yml")
    # copy the meta.yml from dataset to models
    os.system(f"cp ./datasets/{dataset_name}/meta.yml ./models/model_{dataset_name}_{timestamp}_meta.yml")
