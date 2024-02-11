# Encoder-Decoders (own) working from colab

import yaml

import torch
import torch.nn as nn
import torch.optim as optim

# Define Encoder using VGG (you may replace it with your desired encoder)
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
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # size: 128x128 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # size: 64x64
            nn.Flatten(),  # size: 64x64
            nn.Linear(64 * 64 * 64, 4096)  # size: 4096
        )

    def forward(self, x):
        x = self.features(x)
        return x

# Define Decoder using a simple 3-layer MLP
class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Load the ranges from the YAML file
with open('./ranges.yml', 'r') as file:
    ranges = yaml.safe_load(file)

# Create a mapping between parameter names and output sizes
parameter_output_mapping = {}
for param_name, param_specs in ranges.items():
    if param_specs['type'] == 'float':
        parameter_output_mapping[param_name] = 1  # 1 for scalar
    elif param_specs['type'] == 'int':
        parameter_output_mapping[param_name] = 1  # 1 for scalar
    elif param_specs['type'] == 'vector':
        parameter_output_mapping[param_name] = 3  # 3 for x, y, z
    elif param_specs['type'] == 'states':
        parameter_output_mapping[param_name] = len(param_specs['values'])
    elif param_specs['type'] == 'bool':
        parameter_output_mapping[param_name] = 2  # 2 for binary encoding

# Create decoders based on the mapping
decoders = nn.ModuleDict({
    param_name: Decoder(4096, output_size)
    for param_name, output_size in parameter_output_mapping.items()
})

# Complete model with one Pre-trained Encoder and multiple Decoders
class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoders):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoders = decoders

    def forward(self, x):
        x = self.encoder(x)
        print(x.size())
        batch_size = x.size(0)  # Get the batch size
        x = x.view(batch_size, -1)  # Flatten the feature tensor, considering the batch size
        decoder_outputs = {param_name: decoder(x) for param_name, decoder in self.decoders.items()}
        return decoder_outputs

# Example usage
# Create an instance of the EncoderDecoderModel with a pre-trained encoder
encoder = Encoder()  # Use your pre-trained encoder here
model = EncoderDecoderModel(encoder, decoders)

# Example input tensor with shape (batch_size, channels, height, width)
input_tensor = torch.randn(2, 1, 512, 512)

# Forward pass
output = model(input_tensor)

# Print the output shapes for each decoder
for param_name, decoder_output in output.items():
    print(f"{param_name} decoder output shape: {decoder_output.shape}")
