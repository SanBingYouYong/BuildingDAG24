import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms

class SimpleOverfitModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleOverfitModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x


transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Define the input image (just one example)
# input_image = torch.tensor([1.0, 0.0, 1.0, 0.0])  # Example input image (replace with actual image tensor)
input_image = Image.open("datasets/DAGDataset2_1_5/images/batch0_sample0.png")
input_image = transform(input_image).flatten()  # Flatten the image to a 1D tensor

print("Input Image:", input_image)

# Define model, criterion, and optimizer
model = SimpleOverfitModel(input_size=len(input_image), hidden_size=1024)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Using SGD optimizer

# Training loop
for epoch in range(100):  # Increase the number of epochs as needed
    optimizer.zero_grad()
    outputs = model(input_image)
    loss = criterion(outputs, input_image)  # MSE loss between predicted and input image
    loss.backward()
    optimizer.step()
    
    # if (epoch + 1) % 100 == 0:
    print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# After training, check the reconstructed image
reconstructed_image = model(input_image).detach().numpy()
print("Reconstructed Image:", reconstructed_image)
