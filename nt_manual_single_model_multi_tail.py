import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


import os
import yaml
from PIL import Image
from tqdm import tqdm



import torch.nn as nn
import torch.nn.functional as F

class SingleEncoderDecoderModel(nn.Module):
    def __init__(self):
        super(SingleEncoderDecoderModel, self).__init__()
        
        # Encoder layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        
        # Decoder layers for classification
        self.fc_class1 = nn.Linear(64*128*128, 512)
        self.fc_class2 = nn.Linear(512, 2)  # Bm Base Shape
        
        # Decoder layers for regression
        self.fc_reg1 = nn.Linear(64*128*128, 512)
        self.fc_reg2 = nn.Linear(512, 3)  # Bm Size
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # x = F.relu(self.conv3(x))
        x = self.flatten(x)
        
        # Decoder for classification
        x_class = F.relu(self.fc_class1(x))
        x_class = self.fc_class2(x_class)
        
        # Decoder for regression
        x_reg = F.relu(self.fc_reg1(x))
        x_reg = self.fc_reg2(x_reg)
        
        return x_class, x_reg



import os
import yaml
import torch
from PIL import Image
from torchvision import transforms

class SingleTaskDataset(torch.utils.data.Dataset):
    def __init__(self, task_param_names: list, dataset_name: str, datasets_folder: str="./datasets", transform=None, device=None):
        self.task_param_names = task_param_names
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
                param_data = yaml.safe_load(file)
                classification_label = param_data[self.task_param_names[0]]  # Assuming classification label is the first element
                regression_values = param_data[self.task_param_names[1]]    # Assuming regression values follow classification label
            # manual norm TODO: add manual denorm
            param_spec = self.ranges[self.task_param_names[1]]
            regression_values = [(regression_values[i] - param_spec[f'{dim}min']) / (param_spec[f'{dim}max'] - param_spec[f'{dim}min']) for i, dim in enumerate(['x', 'y', 'z'])]
            data.append((image_path, classification_label, regression_values))
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, classification_label, regression_values = self.data[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        classification_label = torch.tensor(classification_label, dtype=torch.long, device=self.device)
        regression_values = torch.tensor(regression_values, dtype=torch.float, device=self.device)
        return image, classification_label, regression_values



def train(model, train_dataloader, val_dataloader, optimizer, criterion, device, epochs=10, save_path='./models/single_encdec.pth'):
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (inputs, classification_labels, regression_labels) in progress_bar:
            inputs = inputs.to(device)
            classification_labels = classification_labels.to(device)
            regression_labels = regression_labels.to(device)

            optimizer.zero_grad()
            classification_outputs, regression_outputs = model(inputs)
            
            # Calculate custom loss
            loss = criterion(classification_outputs, regression_outputs, classification_labels, regression_labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / (i+1):.4f}")

        # Validation
        val_loss = validate(model, val_dataloader, device)  # Implement validate method separately
        unified_val_loss = (val_loss[0] + val_loss[1]) / 2
        
        # Logging
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss / len(train_dataloader):.4f}, Validation Loss: classification: {val_loss[0]:.4f}, regression: {val_loss[1]:.4f}; average val loss: {unified_val_loss:.4f}")
        
        # Save best model
        if unified_val_loss < best_val_loss:
            best_val_loss = unified_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

    print(f"Training finished. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")



def validate(model, dataloader, device):
    model.eval()
    classification_val_loss = 0.0
    regression_val_loss = 0.0

    with torch.no_grad():
        for inputs, classification_labels, regression_labels in dataloader:
            inputs = inputs.to(device)
            classification_labels = classification_labels.to(device)
            regression_labels = regression_labels.to(device)

            # Forward pass
            classification_outputs, regression_outputs = model(inputs)
            
            # Calculate losses separately
            classification_val_loss += F.cross_entropy(classification_outputs, classification_labels).item()
            regression_val_loss += F.mse_loss(regression_outputs, regression_labels).item()

    # Average the losses
    classification_val_loss /= len(dataloader)
    regression_val_loss /= len(dataloader)
    
    # You can return both losses or any other aggregated metric you're interested in
    return classification_val_loss, regression_val_loss


def test(model, test_dataloader, device, save_path='./models/single_encdec_mt_test_results.yml'):
    model.eval()
    classification_predictions = []
    regression_predictions = []
    ground_truth_classification = []
    ground_truth_regression = []
    
    with torch.no_grad():
        for inputs, classification_labels, regression_labels in test_dataloader:
            inputs = inputs.to(device)
            classification_labels = classification_labels.to(device)
            regression_labels = regression_labels.to(device)
            classification_outputs, regression_outputs = model(inputs)
            
            # Classification predictions
            _, classification_predicted = torch.max(classification_outputs, 1)
            classification_predictions.extend(classification_predicted.cpu().numpy().tolist())
            ground_truth_classification.extend(classification_labels.cpu().numpy().tolist())
            
            # Regression predictions
            regression_predictions.extend(regression_outputs.cpu().numpy().tolist())
            ground_truth_regression.extend(regression_labels.cpu().numpy().tolist())
    
    # Pair them 
    results = [
        [classification_predictions[i], regression_predictions[i], ground_truth_classification[i], ground_truth_regression[i]]
        for i in range(len(classification_predictions))
    ]
    
    # Save results
    with open(save_path, 'w') as f:
        yaml.dump({"results": results}, f)

    print(f"Test results saved to {save_path}")



def custom_loss(classification_output, regression_output, classification_target, regression_target, classification_weight=1.0, regression_weight=1.0):
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
    regression_loss = F.mse_loss(regression_output, regression_target)
    
    # Combine the losses
    loss = classification_weight * classification_loss + regression_weight * regression_loss
    
    return loss



if __name__ == "__main__":
    # Instantiate model, dataset, dataloader, optimizer, and criterion

    # Assuming you have defined your model, dataset, optimizer, and criterion
    model = SingleEncoderDecoderModel().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = SingleTaskDataset(task_param_names=['Bm Base Shape', 'Bm Size'], dataset_name='DAGDataset100_100_5')

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
    train(model, train_dataloader, val_dataloader, optimizer, custom_loss, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Test the model
    test(model, test_dataloader, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

