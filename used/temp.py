import torch
import torch.nn as nn
import torch.optim as optim

# Define your model
class YourModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(YourModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x = self.fc(x)
        return x

# Example usage
# Define your model, criterion, and optimizer
input_size = 10  # example input size
num_classes = 5  # example number of classes
model = YourModel(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example data
inputs = torch.randn(3, input_size)  # example batch of input data
targets = torch.randint(0, num_classes, (3,))  # example batch of target labels
print(targets)

# Forward pass
outputs = model(inputs)
outputs = torch.softmax(outputs, dim=1)
print(outputs)

# Calculate loss
loss = criterion(outputs, targets)
print(loss)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()
