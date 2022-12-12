from torchvision import datasets, transforms
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchinfo

class Net(nn.Module):
    def __init__(self, lr=1.):
        super(Net, self).__init__()
        self.conv1 =  nn.Conv2d(3, 16, 3, 1)
        self.batch2dnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.batch2dnorm2 = nn.BatchNorm2d(32)

        self.dense1 = nn.Linear(6272, 50)
        self.dense2 = nn.Linear(50, 10)

        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)


    def forward(self, X):
        y = self.conv1(X)
        y = self.batch2dnorm1(y)
        y = F.relu(y)

        y = self.conv2(y)
        y = self.batch2dnorm2(y)
        y = F.relu(y)
        y = F.max_pool2d(y, 2)

        y = F.relu(self.dense1(y.flatten(1)))
        y = F.relu(self.dense2(y))
        return y #F.softmax(y, dim=1)

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
                print("Epoch: {}, Batch {}, loss {}".format(epoch, batch_id, loss))


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