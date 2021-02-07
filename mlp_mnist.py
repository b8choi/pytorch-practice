import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


num_epochs = 10
batch_size = 256
learning_rate = 0.001


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


model = Net(784, 256, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


train_data = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


def evaluate(model, loader, device):
    cnt = 0
    total = 0
    for x, y in loader:
        x = x.reshape(-1, 784).to(device)
        y = y.to(device)

        output = model(x)
        _, y_pred = torch.max(output, 1)

        cnt += (y_pred == y).sum().item()
        total += len(x)

    print('accuracy:', cnt / total)


def train(model, loader, device):
    for epoch in range(num_epochs):
        for x, y in train_loader:
            x = x.reshape(-1, 784).to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        evaluate(model, test_loader, device)


evaluate(model, test_loader, device)
train(model, train_loader, device)
