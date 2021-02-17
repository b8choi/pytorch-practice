import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from marvel_heroes_dataset import MarvelHeroes


num_epochs = 20
batch_size = 32
learning_rate = 0.001


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(57600, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


train_data = MarvelHeroes(root='./data/', train=True)
test_data = MarvelHeroes(root='./data/', train=False)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


def evaluate(model, loader, device):
    cnt = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        _, y_pred = torch.max(output, 1)

        cnt += (y_pred == y).sum().item()
        total += len(x)

    print('accuracy:', cnt / total)


def train(model, loader, device):
    for epoch in range(num_epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch:', epoch)
        evaluate(model, test_loader, device)


evaluate(model, test_loader, device)
train(model, train_loader, device)
