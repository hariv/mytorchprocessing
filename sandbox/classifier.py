import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch.optim as optim

#Utility function to print image.
def imshow(img):
	img = img/2 + 0.5
	npimg = img.numpy()
	#print(npimg.shape)
	cv2.imshow("image", np.transpose(npimg, (1,2,0)))
	#cv2.waitKey()
	#plt.imshow(np.transpose(npimg, (1, 2, 0)))

class Net(nn.Module):
	#Definition of network.
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	#Forward prop. First pass through conv followed by pool
	#Then fully connected.
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
# The output of torchvision datasets are PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1].
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Load trainset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

#Pass opened trainset through dataloader. trainloader helps me iterate.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
#Same thing with test data.
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

#definition of classes.
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#get a batch of 4 from 
dataiter = iter(trainloader)
images, labels = dataiter.next()

#labels is a number from 0 to 9, representing the 10 classes

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#initialize the neural network.
net = Net()

#define the loss
criterion = nn.CrossEntropyLoss()

#define the optimizer. Could be batch gradient descent, stochastic,
#with or without adam, momentum and all.
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
	running_loss = 0.0

	for i, data in enumerate(trainloader, 0):
		# Take 4 images from the dataset.
		inputs, labels = data
		optimizer.zero_grad()
		#Get the output of these 4 images.
		outputs = net(inputs)
		#compute the loss on these 4 images.
		loss = criterion(outputs, labels)
		#back propagate the loss to get change in weights.
		loss.backward()
		#update weights.
		optimizer.step()
		#compute loss
		running_loss += loss.item()

		#we display loss per 2000.
		#otherwise doesn't have any significance.
		if i % 2000 == 1999:
			print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print("Done training")

dataiter = iter(testloader)
#get one test data batch (will have 4 images).
images, labels = dataiter.next()

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#The outputs are engergies since this is cross entropy loss.
outputs = net(images)

#Get the predictions for this batch (of 4 images).
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0

with torch.no_grad():
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct = (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))