import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


import os
import yaml
from PIL import Image
from tqdm import tqdm


from nn_models import EncDecsLoss
from nn_training import load_metadata



def manual_loss(classification_output, regression_output, classification_target, regression_target, classification_weight=1.0, regression_weight=1.0):
    """
    Custom loss function to handle both classification and regression tasks.

    Args:
        classification_output (torch.Tensor): Predicted output for classification task.
        regression_output (torch.Tensor): Predicted output for regression task.
        classification_target (torch.Tensor): Target for classification task.
        regression_target (torch.Tensor): Target for regression task.
        classification_weight (float): Weight for classification loss (default: 1.0).
        regression_weight (float): Weight for regression loss (default: 1.0).

    Returns:
        torch.Tensor: Combined loss.
    """
    # Classification loss
    classification_loss = F.cross_entropy(classification_output, classification_target)
    
    # Regression loss
    # regression_loss = F.mse_loss(regression_output, regression_target)
    regression_loss = F.l1_loss(regression_output, regression_target)
    
    # Combine the losses
    loss = classification_weight * classification_loss + regression_weight * regression_loss
    
    return loss

def reproduced_loss(outputs, targets):
    # classification: 
    classification_loss = F.cross_entropy(
        outputs["Building Mass Decoder"][0]["Bm Base Shape"], targets["Building Mass Decoder"]["classification_targets"]["Bm Base Shape"]
    )
    # regression:
    regression_loss = F.l1_loss(
        outputs["Building Mass Decoder"][1]["Bm Size"], targets["Building Mass Decoder"]["regression_target"]["Bm Size"]
    )
    return 1 * classification_loss + 1 * regression_loss


if __name__ == "__main__":
    dataset_name = "DAGDataset10_10_5"
    single_decoder = "Building Mass Decoder"

    ranges, parameter_output_mapping, decoders, switches, batch_cam_angles = load_metadata(dataset_name, single_decoder=single_decoder)

    og_loss = EncDecsLoss(decoders, switches, lx_regularizor=-1)


    # generate random data
    classification_output = torch.randn(5, 5)
    regression_output = torch.randn(5, 1)
    classification_target = torch.randint(0, 5, (5,))
    regression_target = torch.randn(5, 1)

    # calculate the loss
    og_loss_val = og_loss(
        {
            "Building Mass Decoder": [
                {
                    "Bm Base Shape": classification_output,
                },
                {
                    "Bm Size": regression_output
                }
            ]
        },
        {
            "Building Mass Decoder": {
                "classification_targets": {
                    "Bm Base Shape": classification_target
                },
                "regression_target": {
                    "Bm Size": regression_target
                }
            }
        }
    )
    print(f"Original Loss: {og_loss_val}")

    manual_loss_val = manual_loss(classification_output, regression_output, classification_target, regression_target)
    print(f"Manual Loss: {manual_loss_val}")
    reproduced_loss_val = reproduced_loss(
        {
            "Building Mass Decoder": [
                {
                    "Bm Base Shape": classification_output,
                },
                {
                    "Bm Size": regression_output
                }
            ]
        },
        {
            "Building Mass Decoder": {
                "classification_targets": {
                    "Bm Base Shape": classification_target
                },
                "regression_target": {
                    "Bm Size": regression_target
                }
            }
        }
    )
    print(f"Reproduced Loss: {reproduced_loss_val}")



