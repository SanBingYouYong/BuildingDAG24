import torch


outputs = torch.tensor(
    [
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    ]
)
target = torch.tensor([4])

print(torch.argmax(outputs, dim=1))
print(torch.softmax(outputs, dim=1))
print(torch.nn.CrossEntropyLoss()(outputs, target))
