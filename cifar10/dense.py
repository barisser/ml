import os

from torchvision import datasets, transforms
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchinfo


class Net(nn.Module):
    def __init__(self, lr=1.0):
        super(Net, self).__init__()
        self.dense1 = nn.Linear(3072, 100)
        self.dense2 = nn.Linear(100, 10)

        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)


    def forward(self, X):
        X = X.flatten(1)
        y = self.dense1(X)
        y = F.relu(y)
        y = F.relu(self.dense2(y))
        return F.softmax(y)

    def fit(self, data_loader, device=torch.device('cpu'), epochs=1):
        self.train()
        for epoch in range(epochs):
            for batch_id, (X, Y) in enumerate(data_loader):
                X, Y = X.to(device), Y.to(device)
                self.optimizer.zero_grad()
                output = self(X)
                loss = F.nll_loss(output, Y)
                loss.backward()
                self.optimizer.step()

                if batch_id % 50 == 0:
                    prediction = output.argmax(dim=1, keepdim=True)
                    correct = prediction.eq(Y.view_as(prediction)).sum().item()
                    percent = round(correct / data_loader.batch_size * 100, 2)
                    print("Epoch: {}, Batch {}, Accuracy: {}, Loss: {}".format(epoch, batch_id, percent, loss))


    def test(self, data_loader, device=torch.device('cpu')):
        model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for X, Y in data_loader:
                X, Y = X.to(device), Y.to(device)
                output = self(X)
                loss += F.nll_loss(output, Y, reduction='sum')
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(Y.view_as(prediction)).sum().item()

        loss /= len(data_loader)
        count = len(data_loader) * data_loader.batch_size
        percent = round(correct / count * 100, 2)
        print("Test: Average loss: {}, Accuracy: {} / {} -> {}".format(
            loss, correct, count, percent))


if __name__ == "__main__":
    model = Net()
    if os.path.exists('dense.model'):
        model.load_state_dict(torch.load('dense.model'))
        
    transform = transforms.ToTensor()
    training_data = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    validation_data = datasets.CIFAR10('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=200, shuffle=True)
    test_loader = torch.utils.data.DataLoader(validation_data, batch_size=200, shuffle=True)
    print(torchinfo.summary(model))

    for i in range(30):
        print(i)
        model.fit(train_loader, epochs=1)
        model.test(train_loader)
        model.test(test_loader)

        torch.save(model.state_dict(), 'dense.model')