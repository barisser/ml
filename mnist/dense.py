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
        self.dense1 = nn.Linear(784, 200)
        self.dense2 = nn.Linear(200, 50)
        self.dense3 = nn.Linear(50, 10)

        self.lr = lr
        self.optimizer = optim.Adadelta(self.parameters(), lr=self.lr)


    def forward(self, X):
        X = X.flatten(1)
        y = self.dense1(X)
        y = F.relu(y)
        y = F.relu(self.dense2(y))
        y = F.relu(self.dense3(y))
        return F.softmax(y, dim=1)

    def fit(self, data_loader, device=torch.device('cpu'), epochs=1, print_results_period=50, outpath=None):
        self.train()
        for epoch in range(epochs):
            for batch_id, (X, Y) in enumerate(data_loader):
                X, Y = X.to(device), Y.to(device)
                self.optimizer.zero_grad()
                output = self(X)
                loss = F.nll_loss(output, Y)
                loss.backward()
                self.optimizer.step()
                if print_results_period and batch_id % print_results_period == 0:
                    prediction = output.argmax(dim=1, keepdim=True)
                    correct = prediction.eq(Y.view_as(prediction)).sum().item()
                    accuracy = round(correct / data_loader.batch_size * 100, 2)
                    print("Epoch: {}, Batch {}, in-sample Accuracy {}, Loss {}".format(epoch, batch_id, accuracy, loss))
            if outpath:
                torch.save(model.state_dict(), outpath)


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

    modelpath = 'mnist_dense.model'
    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath))
        print("Loaded from {}".format(modelpath))

    print(torchinfo.summary(model))
    transform = transforms.ToTensor()
    training_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
    validation_data = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=200)
    test_loader = torch.utils.data.DataLoader(validation_data, batch_size=200)

    epochs = 10
    model.fit(train_loader, epochs=epochs, outpath=modelpath)
    model.test(test_loader)
