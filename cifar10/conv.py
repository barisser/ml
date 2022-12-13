from torchvision import datasets, transforms
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchinfo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device={}".format(device))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 =  nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 8, 3, 1)

        self.dense1 = nn.Linear(288, 100)
        self.dense2 = nn.Linear(100, 10)


    def forward(self, X):
        y = self.conv1(X)
        y = F.relu(y)
        y = F.max_pool2d(y, 2)
        
        y = self.conv2(y)
        y = F.relu(y)
        y = F.max_pool2d(y, 2)

        y = torch.tanh(self.dense1(y.flatten(1)))
        y = self.dense2(y)
        return y
    
    def predict(self, x):
        y = self(x)
        return torch.max(y, dim=1)

    def test(self, data_loader, device=device):
        model.eval()
        loss = 0
        correct = 0
        with torch.no_grad():
            for X, Y in data_loader:
                X, Y = X.to(device), Y.to(device)
                output = self(X)
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(Y.view_as(prediction)).sum().item()

        count = len(data_loader) * data_loader.batch_size
        percent = round(correct / count * 100, 2)
        print("Test: Accuracy: {} / {} -> {}".format(
            correct, count, percent))

        

if __name__ == "__main__":
    model = Net().to(device=device)
    if torch.cuda.is_available():
        model.cuda()
        
    transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))])

    training_data = datasets.CIFAR10('../data', train=True, download=True, transform=transforms)
    validation_data = datasets.CIFAR10('../data', train=False, transform=transforms)
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=200, shuffle=True)
    test_loader = torch.utils.data.DataLoader(validation_data, batch_size=200, shuffle=True)
    print(torchinfo.summary(model))


    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    n_epochs = 100
    loss_func = nn.CrossEntropyLoss().to(device=device)
    for epoch in range(n_epochs):
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device=device), labels.to(device=device)
            outputs = model(imgs)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.test(test_loader)
        print("Epoch: %d, Loss: %f" % (epoch, float(loss)))
    
    model.test(test_loader)

