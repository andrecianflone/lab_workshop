
import torch
import torch.nn as nn
import torch.optim as optim

train_dataset = MNIST(root='../data',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)

test_dataset = MNIST(root='../data',
                     train=False,
                     transform=transforms.ToTensor())

batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

use_gpu = torch.cuda.is_available()

import torch.nn as nn

input_size = 784
hidden_size = 500
num_classes = 10

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, num_classes))

    def forward(self, x):

        out = self.hidden_layer(x)

        out = self.output_layer(out)

        return out

model = MLP(input_size, hidden_size, num_classes)

if use_gpu:
  # switch model to GPU
  model.cuda()

print(model)
import torch.nn as nn

input_size = 784
hidden_size = 500
num_classes = 10

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, num_classes))

    def forward(self, x):

        out = self.hidden_layer(x)

        out = self.output_layer(out)

        return out

model = MLP(input_size, hidden_size, num_classes)

if use_gpu:
  # switch model to GPU
  model.cuda()

print(model)
print("# parameters: ", sum([param.nelement() for param in model.parameters()]))

learning_rate = 1e-3

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 1
for epoch in range(num_epochs):

    train_loss = 0
    n_iter = 0

    # Iterate over data.
    for i, (images, labels) in enumerate(train_loader):

        if use_gpu:
          # switch tensor type to GPU
          images = images.cuda()
          labels = labels.cuda()

        # Flatten the images
        images = images.view(-1, 28*28)

        # Zero the gradient buffer
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        loss = criterion(outputs, labels)
        total_loss.append(loss)
        # Backward
        loss.backward()

        # Optimize
        optimizer.step()

        # Statistics
        train_loss += loss.data[0]
        n_iter += 1

    print('Epoch: {}/{}, Loss: {:.4f}'.format(
          epoch+1, num_epochs, train_loss/n_iter))

# Set model to evaluate mode
model.eval()

correct = 0
total = 0

# Iterate over data.
for images, labels in test_loader:

    if use_gpu:
      # switch tensor type to GPU
      images = images.cuda()
      labels = labels.cuda()

    # Flatten the images
    images = images.view(-1, 28*28)

    # Forward
    outputs = model(images)
    loss = criterion(outputs, labels)
    _, predicted = torch.max(outputs.data, 1)

    # Statistics
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy on the test set: {}%'.format(100 * correct / total))





