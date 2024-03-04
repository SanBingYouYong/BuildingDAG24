import yaml
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # size: 512x512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # size: 256x256

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # size: 256x256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # size: 128x128

            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # size: 128x128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # size: 64x64

            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 1024),  # size: 1024
        )

    def forward(self, x):
        x = self.features(x)
        return x


class ParamAwareMultiTailDecoder(nn.Module):
    def __init__(self, input_size, classification_params=None, regression_params=None, dropout_prob=0.5):
        super(ParamAwareMultiTailDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(p=dropout_prob)
        self.classification_tails = nn.ModuleDict(
            {
                param_name: nn.Sequential(
                    nn.Linear(512, size),
                    # nn.Softmax(dim=1),
                )
                for param_name, size in classification_params.items()
            }
        ) if classification_params else {}

        self.regression_tail = nn.ModuleDict(
            {
                param_name: nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    # nn.Dropout(p=dropout_prob),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    # nn.Dropout(p=dropout_prob),
                    nn.Linear(128, size),
                )
                for param_name, size in regression_params.items()
            }
        ) if regression_params else {}

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        # x = self.dropout1(x)
        classification_outputs = {
            param_name: tail(x) for param_name, tail in self.classification_tails.items()
        } if self.classification_tails else {}
        regression_output = {
            param_name: tail(x) for param_name, tail in self.regression_tail.items()
        } if self.regression_tail else {}
        return classification_outputs, regression_output


class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoders):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoders = decoders

    def forward(self, x):
        x = self.encoder(x)
        batch_size = x.size(0)  # Get the batch size
        x = x.view(batch_size, -1)  # Flatten the feature tensor, considering the batch size
        decoder_outputs = {decoder_name: decoder(x) for decoder_name, decoder in self.decoders.items()}
        return decoder_outputs  # note that the multi-tail decoder returns a list of outputs


# Define loss function and optimizer
# for regression, use MSELoss (or L1), for classification, use CrossEntropyLoss
class EncDecsLoss(nn.Module):
    def __init__(self, decoders, switches_mapping: dict):
        super(EncDecsLoss, self).__init__()
        self.decoders = decoders
        self.switches_mapping = switches_mapping

    def forward(self, outputs, targets):
        loss = 0.0
        for decoder_name, decoder_output in outputs.items():
            cur_decoder_loss = self.decoder_loss(decoder_output, targets[decoder_name])
            print(decoder_name, cur_decoder_loss.item())
            loss += cur_decoder_loss
        loss /= len(outputs)
        return loss

    def classification_loss(self, output, target):
        # output = torch.tensor([[1., 0.]]).to(target.device)
        print(f" - Classification output: {output}, target: {target}")
        # loss = nn.CrossEntropyLoss()(output, target)
        loss = nn.CrossEntropyLoss()(torch.argmax(output).float().unsqueeze(0), torch.argmax(target).float().unsqueeze(0))
        # print(f"--{torch.argmax(output, dim=1)}")
        # print(f"--loss: {loss}")
        # print(f"~~~{torch.argmax(output).float().unsqueeze(0)}")
        # print(f"~~~{torch.argmax(target).float().unsqueeze(0)}")
        # print(f"~~~loss: {nn.CrossEntropyLoss()(torch.argmax(output).float().unsqueeze(0), torch.argmax(target).float().unsqueeze(0))}")
        # raise
        # loss = nn.CrossEntropyLoss()(output.argmax(dim=1), target)
        # loss = F.cross_entropy(output, target)
        print(f" - - Calculated loss: {loss}")
        return loss

    def regression_loss(self, output, target):
        print(f" - Regression output: {output}, target: {target}")
        # return nn.MSELoss()(output, target)
        loss = nn.L1Loss()(output, target)
        print(f" - - Calculated loss: {loss}")
        return loss

    def decoder_loss(self, decoder_output, target):
        classification_outputs = decoder_output[0]  # note that model outputs a tuple of list instead of dict of list
        regression_output = decoder_output[1]
        total_classification_loss = 0.0
        # if classification_outputs:
        for param_name, pred in classification_outputs.items():
            total_classification_loss += self.classification_loss(pred, target["classification_targets"][param_name])
            # TODO: should we add early termination for "Has" labels?
        # if regression_output:
        total_regression_loss = 0.0
        for param_name, pred in regression_output.items():
            regression_loss = self.regression_loss(pred, target["regression_target"][param_name])
            # use gt's 0 1 label to switch off the loss if needed
            switch_param_name = self.switches_mapping["Reversed Mapping"].get(param_name)
            if switch_param_name:
                # print(f"Switching off {param_name} loss")
                # print(f"Switch: {switch_param_name}")
                # print(f"Original Regression Loss: {regression_loss}")
                switch_target = target["classification_targets"][switch_param_name]
                # print("!!Switch Target:", switch_target)
                # switch_index = torch.argmin(switch_target, dim=1)
                switch_index = switch_target
                # print("!!Switch Index:", switch_index)
                # make regression_loss same shape as switch_index
                regression_loss = torch.stack([regression_loss] * switch_index.size(0))
                regression_loss *= switch_index
                # average again
                regression_loss = torch.mean(regression_loss)
                # print(f"New Regression Loss: {regression_loss}")
            total_regression_loss += regression_loss
        print(f"Classification Loss: {total_classification_loss}, Regression Loss: {total_regression_loss}")
        averaged_classification_loss = total_classification_loss / len(classification_outputs) if len(classification_outputs) > 0 else 0
        averaged_regression_loss = total_regression_loss / len(regression_output) if len(regression_output) > 0 else 0
        print(f"!! Averaged Classification Loss: {averaged_classification_loss}, Regression Loss: {averaged_regression_loss}")
        loss = averaged_classification_loss + averaged_regression_loss
        return loss


if __name__ == "__main__":
    from nn_dataset import DAGDataset, split_dataset, overfit_dataloaders
    from nn_training import train, load_metadata

    # give EncDecLoss identical outputs and targets, see if it returns 0 loss
    
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

    # get one image from the dataset
    sample, target = train_dataset[0]
    sample = sample.unsqueeze(0).to(device)
    target = {decoder_name: {task: {param_name: param.unsqueeze(0).to(device) for param_name, param in task_params.items()} for task, task_params in decoder_outputs.items()} for decoder_name, decoder_outputs in target.items()}
    # print("Sample:", sample)
    print("!!!Target:", target)
    # raise

    # feed to model
    outputs = model(sample)
    print("!!!Outputs:", outputs)

    loss = criterion(outputs, target)
    print("\n!!!Loss:", loss)
    # raise

    # copy paste target values to corresponding entries in outputs
    for decoder_name, (classification_outputs, regression_outputs) in outputs.items():
        # print(decoder_name)    
        for param_name, param in classification_outputs.items():
            classification_outputs[param_name] = target[decoder_name]["classification_targets"][param_name]
            # convert class indices to one-hot
            # classification_outputs[param_name] = F.one_hot(target[decoder_name]["classification_targets"][param_name], num_classes=param.size(1)).float()
        for param_name, param in regression_outputs.items():
            regression_outputs[param_name] = target[decoder_name]["regression_target"][param_name]
    # print("Outputs with Target Values:", outputs)

    # # check that now outputs and targets are identical
    # for decoder_name, (classification_outputs, regression_outputs) in outputs.items():
    #     for param_name, param in classification_outputs.items():
    #         assert torch.allclose(param, target[decoder_name]["classification_targets"][param_name])
    #     for param_name, param in regression_outputs.items():
    #         assert torch.allclose(param, target[decoder_name]["regression_target"][param_name])

    # calculate loss
    loss = criterion(outputs, target)
    print("Loss:", loss)
