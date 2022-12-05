from torchvision import datasets, transforms
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F


transform = transforms.ToTensor()
training_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
validation_data = datasets.MNIST('../data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(training_data, batch_size=200)
test_loader = torch.utils.data.DataLoader(validation_data, batch_size=200)

class Net(nn.Module):
	def __init__(self, lr=0.01):
		super(Net, self).__init__()
		"""
		convolute and relu twice
		max pool in 2d
		2 dense layers
		"""
		self.conv1 = nn.Conv2d(1, 4, 3, 1)
		self.conv2 = nn.Conv2d(4, 12, 3, 1)

		self.dense1 = nn.Linear(1728, 100)
		self.dense2 = nn.Linear(100, 10)

		self.lr = lr
		self.optimizer = optim.Adadelta(self.parameters(), lr=self.lr)


	def forward(self, X):
		y = F.relu(self.conv1(X))
		y = F.relu(self.conv2(y))
		y = F.max_pool2d(y, 2)
		y2 = torch.flatten(y, 1)
		y2 = F.relu(self.dense1(y2))
		y2 = F.relu(self.dense2(y2))
		return F.softmax(y2, dim=1)

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
					print("Epoch: {}, Batch {}".format(epoch, batch_id))


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

	epochs = 20
	for _ in range(epochs):
		model.fit(train_loader, epochs=1)
		model.test(test_loader)

