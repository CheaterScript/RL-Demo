import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transform pipe
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define network model


class NumberClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flattan = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28,  128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flattan(x)
        x = self.layers(x)
        return x

# Define train function


def train(model: NumberClassifier, dataLoader):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    optimizer = optim.Adam(model.parameters(), 0.001)
    loss_function = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(dataLoader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = running_loss / 100
            accuracy = 100. * correct / total
            print(f'[{batch_idx * 64}/{60000}]'
                  f'Loss: {avg_loss:.3f}  |  Accuracy: {accuracy: .1f}%'
                  )
            running_loss = 0


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100. * correct / total


model = NumberClassifier().to(device=device)

num_epochs = 10
for epoch in range(num_epochs):
    print(f'\nEpoch: {epoch + 1}')
    train(model=model, dataLoader=train_loader)
    accuracy = evaluate(model=model, test_loader=test_loader)
    print(f'Test Accuracy: {accuracy:.2f}%')
