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
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # size: 512x512
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # size: 256x256

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # size: 256x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # size: 128x128

            # nn.Conv2d(32, 32, kernel_size=3, padding=1),  # size: 128x128
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),  # size: 64x64

            nn.Flatten(),
            nn.Linear(64 * 128 * 128, 1024),  # size: 1024
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.features(x)
        return x


class ParamAwareMultiTailDecoder(nn.Module):
    def __init__(self, input_size, classification_params=None, regression_params=None, dropout_prob=0.5):
        super(ParamAwareMultiTailDecoder, self).__init__()
        # self.fc1 = nn.Linear(input_size, 1024)
        # self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(p=dropout_prob)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(p=dropout_prob)
        self.classification_tails = nn.ModuleDict(
            {
                param_name: nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.ReLU(),
                    # nn.Dropout(p=dropout_prob),
                    # nn.Linear(512, 256),
                    # nn.ReLU(),
                    # nn.Dropout(p=dropout_prob),
                    # nn.Linear(256, 128),
                    # nn.ReLU(),
                    # nn.Dropout(p=dropout_prob),
                    nn.Linear(512, size),
                )
                for param_name, size in classification_params.items()
            }
        ) if classification_params else {}

        self.regression_tail = nn.ModuleDict(
            {
                param_name: nn.Sequential(
                    # nn.Linear(1024, 1024),
                    # nn.ReLU(),
                    # nn.Dropout(p=dropout_prob),
                    nn.Linear(input_size, 512),
                    nn.ReLU(),
                    # nn.Dropout(p=dropout_prob),
                    # nn.Linear(512, 512), #
                    # nn.ReLU(), #
                    # nn.Dropout(p=dropout_prob), #
                    # nn.Linear(512, 256), #
                    # nn.ReLU(), #
                    # nn.Dropout(p=dropout_prob), #
                    # nn.Linear(256, 128), #
                    # nn.ReLU(), #
                    # nn.Dropout(p=dropout_prob), #
                    nn.Linear(512, size), #
                )
                for param_name, size in regression_params.items()
            }
        ) if regression_params else {}

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.relu1(x)
        # x = self.dropout1(x)
        # x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.dropout2(x)
        classification_outputs = nn.ModuleDict({
            param_name: tail(x) for param_name, tail in self.classification_tails.items()
        }) if self.classification_tails else {}
        regression_output = nn.ModuleDict({
            param_name: tail(x) for param_name, tail in self.regression_tail.items()
        }) if self.regression_tail else {}
        return classification_outputs, regression_output


class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoders):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoders = decoders

    def forward(self, x):
        x = self.encoder(x)
        decoder_outputs = {decoder_name: decoder(x) for decoder_name, decoder in self.decoders.items()}
        return decoder_outputs  # note that the multi-tail decoder returns a list of outputs


class ManualEncoderDecoderModelBM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 128 * 128, 1024)
        self.relu3 = nn.ReLU()

        self.cls_fc1 = nn.Linear(1024, 512)
        self.cls_relu1 = nn.ReLU()
        self.cls_fc2 = nn.Linear(512, 2)

        self.reg_fc1 = nn.Linear(1024, 512)
        self.reg_relu1 = nn.ReLU()
        self.reg_fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)

        cls_x = self.cls_fc1(x)
        cls_x = self.cls_relu1(cls_x)
        cls_x = self.cls_fc2(cls_x)

        reg_x = self.reg_fc1(x)
        reg_x = self.reg_relu1(reg_x)
        reg_x = self.reg_fc2(reg_x)

        return {"Building Mass Decoder": ({"Bm Base Shape": cls_x}, {"Bm Size": reg_x})}


def custom_loss(outputs, targets):
    # classification: 
    classification_loss = F.cross_entropy(
        outputs["Building Mass Decoder"][0]["Bm Base Shape"], targets["Building Mass Decoder"]["classification_targets"]["Bm Base Shape"]
    )
    # regression:
    regression_loss = F.l1_loss(
        outputs["Building Mass Decoder"][1]["Bm Size"], targets["Building Mass Decoder"]["regression_target"]["Bm Size"]
    )
    return 1 * classification_loss + 1 * regression_loss


# Define loss function and optimizer
# for regression, use MSELoss (or L1), for classification, use CrossEntropyLoss
class EncDecsLoss(nn.Module):
    def __init__(self, decoders, switches_mapping: dict, lx_lambda=0.01, lx_regularizor=1):
        '''
        lx is disabled for now. -1 for no
        '''
        super(EncDecsLoss, self).__init__()
        self.decoders = decoders
        self.switches_mapping = switches_mapping
        self.lx_lambda = lx_lambda
        self.lx = lx_regularizor

    # def forward(self, outputs, targets, print_in_val=False):
    def forward(self, outputs, targets):
        loss = 0.0
        for decoder_name, decoder_output in outputs.items():
            loss += self.decoder_loss(decoder_output, targets[decoder_name])
        if self.lx == -1:
            return loss
        lx_reg = 0
        for param in self.parameters():  # TODO: test what is this for loop iterating over
            lx_reg += param.norm(self.lx)
        loss += self.lx_lambda * lx_reg
        return loss

    def classification_loss(self, output, target):
        # loss = nn.CrossEntropyLoss()(output, target)
        loss = F.cross_entropy(output, target)
        return loss

    def regression_loss(self, output, target):
        # return nn.MSELoss()(output, target)
        # check for shape [x, 1] and [x]
        if len(output.size()) == 2 and len(target.size()) == 1 and output.size(1) == 1:
            target = target.unsqueeze(1)
        # return nn.L1Loss()(output, target)
        # return nn.MSELoss()(output, target)
        loss = F.l1_loss(output, target)
        return loss

    def decoder_loss_0(self, decoder_output, target, print_in_val=False):
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
                switch_target = target["classification_targets"][switch_param_name]
                # switch_index = torch.argmin(switch_target, dim=1)
                switch_index = switch_target
                # make regression_loss same shape as switch_index
                regression_loss = torch.stack([regression_loss] * switch_index.size(0))
                regression_loss *= switch_index
                # average again
                regression_loss = torch.mean(regression_loss)
            total_regression_loss += regression_loss
        averaged_classification_loss = total_classification_loss / len(classification_outputs) if len(classification_outputs) > 0 else 0
        averaged_regression_loss = total_regression_loss / len(regression_output) if len(regression_output) > 0 else 0
        # if print_in_val:
        #     print(f"Classification Loss: {averaged_classification_loss}, Regression Loss: {averaged_regression_loss}")
        loss = averaged_classification_loss + averaged_regression_loss
        return loss

    def decoder_loss(self, decoder_output, target, cls_weight=1.0, reg_weight=1.0):        
        classification_outputs = decoder_output[0]
        regression_output = decoder_output[1]
        classification_targets = target["classification_targets"]
        regression_targets = target["regression_target"]

        total_classification_loss = 0.0
        total_regression_loss = 0.0

        for param_name, pred in classification_outputs.items():
            total_classification_loss += self.classification_loss(pred, classification_targets[param_name])
        
        for param_name, pred in regression_output.items():
            total_regression_loss += self.regression_loss(pred, regression_targets[param_name])
        
        loss = cls_weight * total_classification_loss + reg_weight * total_regression_loss

        return loss
