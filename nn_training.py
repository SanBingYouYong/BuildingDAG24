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
# from nn_models_small import *
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

def load_decoders(metadata: dict, ranges: dict, parameter_output_mapping: dict, decoder_input_size=1024, single_decoder: str=None):
    decoders_params = metadata['decoders']
    decoders = nn.ModuleDict()
    # initialize decoders with corresponding output tails
    decoder_names = list(decoders_params.keys())
    with tqdm(total=len(decoder_names), desc='Loading Decoders') as pbar:
        for decoder_name, param_names in decoders_params.items():
            if single_decoder and decoder_name != single_decoder:
                pbar.update(1)
                continue
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
            pbar.update(1)
    return decoders

def load_metadata(dataset_name: str, datasets_folder: str="./datasets", single_decoder: str=None):
    dataset_path = os.path.join(datasets_folder, dataset_name)
    metadata_file_path = os.path.join(dataset_path, "meta.yml")
    with open(metadata_file_path, 'r') as file:
        metadata = yaml.safe_load(file)
    ranges, parameter_output_mapping = load_ranges(metadata)
    decoders = load_decoders(metadata, ranges, parameter_output_mapping, single_decoder=single_decoder)
    switches = metadata['switches']
    batch_cam_angles = metadata['batch_cam_angles']
    return ranges, parameter_output_mapping, decoders, switches, batch_cam_angles

def load_metadata_for_inference(metadata_file_path: str, need_full: bool=False, single_decoder: str=None):
    with open(metadata_file_path, 'r') as file:
        metadata = yaml.safe_load(file)
    ranges, parameter_output_mapping = load_ranges(metadata)
    decoders = load_decoders(metadata, ranges, parameter_output_mapping, single_decoder=single_decoder)
    if not need_full:
        return ranges, parameter_output_mapping, decoders
    else:
        switches = metadata['switches']
        batch_cam_angles = metadata['batch_cam_angles']
        return ranges, parameter_output_mapping, decoders, switches, batch_cam_angles
    

def train(model: nn.Module, criterion: nn.Module, optimizer, train_loader, val_loader, 
          epochs=25, seed=-1, 
          model_save_path="./models/EncDecModel.pth", 
          loss_save_path="./models/loss.yml"):
    '''
    seed is manually set at main. -1 for no reset, others for seed
    '''
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Initialize with a very high value

    if seed != -1:
        torch.manual_seed(seed)

    validation_interval = 100  # Run validation every 10(or 100) batches
    validation_counter = 0

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

            validation_counter += 1
            if validation_counter == validation_interval:
                validation_counter = 0
                
                # Validation
                model.eval()
                val_loss = 0.0
                num_batches_val = len(val_loader)

                with torch.no_grad():
                    for j, val_data in enumerate(val_loader):
                        val_inputs, val_targets = val_data
                        val_outputs = model(val_inputs)
                        val_loss += criterion(val_outputs, val_targets).item()

                val_loss /= num_batches_val
                val_losses.append(val_loss)

                cur_train_loss = train_loss / (i + 1)
                train_losses.append(cur_train_loss)

                print(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{num_batches}, Training Loss: {train_loss / (i + 1)}, Validation Loss: {val_loss}")

                # Save the model if validation loss improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_save_path)

                model.train()
    print("Finished Training")
    print(f"Best Validation Loss: {best_val_loss}; at iteration {val_losses.index(best_val_loss) + 1}")

    # Save the loss
    with open(loss_save_path, "w") as f:
        yaml.dump({"train_losses": train_losses, "val_losses": val_losses}, f)


def parse_outputs(outputs: dict, ranges: dict, targets: dict, idx=0):
    parsed_outputs = {}
    for decoder_name, decoder_outputs in outputs.items():
        # print(f"Decoder: {decoder_name}")
        classification_outputs, regression_output = decoder_outputs
        classification_targets = targets[decoder_name]['classification_targets']
        regression_targets = targets[decoder_name]['regression_target']
        for param_name, pred in classification_outputs.items():
            tar = classification_targets[param_name]
            parsed_pred = []
            for i in range(len(pred)):
                p = denormalize(pred[i], ranges[param_name])
                t = denormalize(tar[i], ranges[param_name], is_target=True)
                # if param_name == "Bm Base Shape":
                #     print(f"{param_name} Param Type: {ranges[param_name]['type']}")
                #     print(f"Model Output: {pred[i]}")
                #     print(f"Target: {tar[i]}")
                #     print(f"deNormalized Model Output: {p}")
                #     print(f"deNormalized Target: {t}")
                #     raise
                parsed_pred.append([p, t])
            parsed_outputs[param_name] = parsed_pred
        for param_name, pred in regression_output.items():
            reg = regression_targets[param_name]
            parsed_pred = []
            for i in range(len(pred)):
                p = denormalize(pred[i], ranges[param_name])
                t = denormalize(reg[i], ranges[param_name], is_target=True)  # although no diff for reg targets
                # if param_name == "Bm Size":
                #     print(f"{param_name} Param Type: {ranges[param_name]['type']}")
                #     print(f"Model Output: {pred[i]}")
                #     print(f"Target: {reg[i]}")
                #     print(f"deNormalized Model Output: {p}")
                #     print(f"deNormalized Target: {t}")
                #     raise
                parsed_pred.append([p, t])
            parsed_outputs[param_name] = parsed_pred
    return parsed_outputs

def test(model: nn.Module, best_weight_path: str, test_loader, criterion: nn.Module, ranges, results_save_path="results.yml"):
    '''
    Saves test results to a file. 
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(best_weight_path, map_location=device))
    model.eval()
    test_loss = 0.0
    num_batches = len(test_loader)

    results = {}

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, targets = data

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            parsed_res = parse_outputs(outputs, ranges, targets)  # contains pred and tar

            # accumulate the last batch of results
            for param_name, parsed_pred in parsed_res.items():
                if param_name not in results:
                    results[param_name] = []
                results[param_name] += parsed_pred

    print(f"Test Loss: {test_loss / num_batches}")

    # save the results
    with open(results_save_path, "w") as f:
        yaml.dump({"results": results}, f)
    return results


if __name__ == "__main__":
    dataset_name = "DAGDataset100_100_5"
    if not os.path.exists(f"./datasets/{dataset_name}"):
        raise FileNotFoundError(f"Dataset {dataset_name} not found")
    
    torch.manual_seed(0)
    
    single_decoder = None
    
    ranges, parameter_output_mapping, decoders, switches, batch_cam_angles = load_metadata(dataset_name, single_decoder=single_decoder)

    dataset = DAGDatasetSingleDecoder(single_decoder, dataset_name) if single_decoder else DAGDataset(dataset_name)
    print(f"Dataset: {dataset_name}")
    print(f"Loaded {len(decoders)} decoders")

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.8, 0.1, 0.1)
    train_loader, val_loader, test_loader = create_dataloaders_of(train_dataset, val_dataset, test_dataset, batch_size=32)
    
    # train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.5, 0.5, 0)
    # train_loader, val_loader = overfit_dataloaders(train_dataset, val_dataset, batch_size=32)
    
    print(f"Train/Val/Test: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")

    encoder = Encoder()
    model = EncoderDecoderModel(encoder, decoders)

    criterion = EncDecsLoss(decoders, switches, lx_regularizor=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    os.makedirs("./models", exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model_name = f"./models/model_{dataset_name}_{timestamp}.pth"
    loss_name = f"./models/model_{dataset_name}_{timestamp}_loss.yml"
    train(model, criterion, optimizer, train_loader, val_loader, epochs=25, seed=-1, model_save_path=model_name, loss_save_path=loss_name)
    test(model, model_name, test_loader, criterion, ranges, results_save_path="results.yml")
    # copy the meta.yml from dataset to models
    os.system(f"cp ./datasets/{dataset_name}/meta.yml ./models/model_{dataset_name}_{timestamp}_meta.yml")

