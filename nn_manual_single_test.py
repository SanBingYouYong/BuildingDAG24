import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


import os
import yaml
from PIL import Image
from tqdm import tqdm



'''
1. Define a Encoder-Decoder network
2. Define a dataset class of input image and output bm base shape
3. Train the model, test performance
'''

class SingleEncoderDecoderModel(nn.Module):
    def __init__(self):
        super(SingleEncoderDecoderModel, self).__init__()
        
        # Encoder layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        
        # Decoder layers
        self.fc1 = nn.Linear(128*128*128, 512)
        self.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        
        # Decoder
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SingleTaskDataset(torch.utils.data.Dataset):
    def __init__(self, task_param_name: str, 
                 dataset_name: str, 
                 datasets_folder: str="./datasets", 
                 transform=None, device=None):
        self.task_param_name = task_param_name
        self.dataset_name = dataset_name
        self.datasets_folder = datasets_folder
        self.dataset_path = os.path.join(self.datasets_folder, self.dataset_name)
        self.images_folder = os.path.join(self.dataset_path, "images")
        self.params_folder = os.path.join(self.dataset_path, "params")
        self.metadata_file_path = os.path.join(self.dataset_path, "meta.yml")
        self.ranges = None
        self.decoders = None
        self.transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ) if transform is None else transform
        self.data = self.load_data()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

    def load_data(self):
        # load metadata
        with open(self.metadata_file_path, 'r') as file:
            metadata = yaml.safe_load(file)
        self.ranges = metadata['ranges']
        self.decoders = metadata['decoders']
        # read images and parameters
        data = []
        for image_name in os.listdir(self.images_folder):
            image_path = os.path.join(self.images_folder, image_name)
            param_path = os.path.join(self.params_folder, os.path.splitext(image_name)[0] + ".yml")
            with open(param_path, 'r') as file:
                param = yaml.safe_load(file)[self.task_param_name]
            data.append((image_path, param))
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, param = self.data[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        param = torch.tensor(param, dtype=torch.long, device=self.device)
        return image, param


def train(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs=10, save_path='./models/single_encdec.pth'):
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / (i+1):.4f}")

        # Validation
        val_loss = validate(model, val_dataloader, criterion, device)
        
        # Logging
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss / len(train_dataloader):.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

    print(f"Training finished. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")


def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    return val_loss / len(dataloader)

def test(model, test_dataloader, device, save_path='./models/single_encdec_test_results.yml'):
    model.eval()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy().tolist())
            ground_truth.extend(labels.cpu().numpy().tolist())
    
    # pair them
    results = [
        [predictions[i], ground_truth[i]] for i in range(len(predictions))
    ]
    
    # Save results
    with open(save_path, 'w') as f:
        # yaml.dump({"predictions": predictions, "ground_truth": ground_truth}, f)
        yaml.dump({"results": results}, f)

    print(f"Test results saved to {save_path}")


if __name__ == "__main__":
    # Instantiate model, dataset, dataloader, optimizer, and criterion

    # Assuming you have defined your model, dataset, optimizer, and criterion
    model = SingleEncoderDecoderModel().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = SingleTaskDataset(task_param_name='Bm Base Shape', dataset_name='DAGDataset100_100_5')

    # Split the dataset into training, validation, and test subsets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = test_size = (len(dataset) - train_size) // 2  # Remaining 10% each for validation and test

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders for training, validation, and test
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, train_dataloader, val_dataloader, optimizer, criterion, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Test the model
    test(model, test_dataloader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

