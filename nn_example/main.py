import torch
import torch.optim as optim
import torch.nn as nn
from util import get_data
from model import MLP
from torch.autograd import Variable

# Some hparams
batch_size  = 100
input_size  = 784
hidden_size = 200
num_classes = 10
num_epochs = 20
learning_rate = 1e-3

def train(train_loader, test_loader, optimizer, model, criterion):
    for epoch in range(num_epochs):

        # TRAIN
        train_loss = 0
        n_iter = 0

        # Iterate over data.
        for images, labels in train_loader:

            images, labels = Variable(images), Variable(labels)
            # Flatten the images
            images = images.view(-1, 28*28)

            # Zero the gradient buffer
            optimizer.zero_grad()

            # Forward
            outputs = model(images)

            loss = criterion(outputs, labels)
            # Backward
            loss.backward()

            # Optimize
            optimizer.step()

            # Statistics
            train_loss += loss.data[0]
            n_iter += 1

        print('Epoch: {}/{}, Loss: {:.4f}'.format(
              epoch+1, num_epochs, train_loss/n_iter))

        # TEST
        model.eval()
        correct = 0
        total = 0

        for images, labels in test_loader:

            images, labels = Variable(images), Variable(labels)
            # Flatten the images
            images = images.view(-1, 28*28)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            # Statistics
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

        print('Accuracy on the test set: {}%'.format(100 * correct / total))


if __name__=='__main__':
    # Get data
    train_loader, test_loader = get_data(batch_size)
    # Get model
    model = MLP(input_size, hidden_size, num_classes)
    print(model)
    print("# parameters: ", sum([param.nelement() for param in model.parameters()]))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train(train_loader, test_loader, optimizer, model, criterion)



