import torch
import torch.nn as nn
import torch.optim as optim


input_dim = 1
output_dim = 1
n_epochs = 1000
learning_rate = 0.01


x_train = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=torch.float32)
y_train = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16], [18], [20]], dtype=torch.float32)


model = nn.Linear(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(n_epochs):
    output = model(x_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(loss)

print(output)
