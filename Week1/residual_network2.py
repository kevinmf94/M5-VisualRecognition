import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
#from torch.utils.tensorboard import SummaryWritter
#import numpy as np 

# Writer will output to ./runs/ directory by default
#writer = SummaryWriter()

# Constants
ENABLE_GPU = True
GPU_ID = 0
#DATASET_TRAIN = '/Users/kevinmartinfernandez/Workspace/Master/M3/BagOfWords/MIT_split/train/'
#DATASET_TEST = '/Users/kevinmartinfernandez/Workspace/Master/M3/BagOfWords/MIT_split/test/'
DATASET_TRAIN = '/home/mcv/datasets/MIT_split/train'
DATASET_TEST = '/home/mcv/datasets/MIT_split/test'
EPOCHS = 300
batch_size = 200
img_width = 32
img_height = 32

# Set GPU
if ENABLE_GPU:
    torch.cuda.set_device(device=GPU_ID)

# Dataset transform (Resize image to 128x128 + Convert ToTensor)
transform = transforms.Compose({
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor()
})

# Load train and test datasets.
trainDataset = datasets.ImageFolder(DATASET_TRAIN, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=int(len(trainDataset) / batch_size), shuffle=True)

testDataset = datasets.ImageFolder(DATASET_TEST, transform=transform)
testLoader = torch.utils.data.DataLoader(testDataset, shuffle=True)


# Define Model
class UnitRestNet(nn.Module):

    def __init__(self, input_dim, filters, kernel=3):
        super(UnitRestNet, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, filters, kernel, 1, 1)
        self.relu = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm2d(filters)
        self.conv3 = nn.Conv2d(filters, filters, kernel, 1, 1)
        self.batchNorm2 = nn.BatchNorm2d(filters)
        self.conv4 = nn.Conv2d(filters, filters, kernel, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        res = x
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = torch.add(res, x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.unitRest1 = UnitRestNet(3, 8, 3)
        self.unitRest2 = UnitRestNet(8, 10, 3)
        self.batchNorm2 = nn.BatchNorm2d(10)
        self.avgPool = nn.AvgPool2d(4, stride=None)
        self.relu = nn.ReLU()
        self.inputsFc1 = 10 * (img_width//4) * (img_height//4)
        self.fc1 = nn.Linear(self.inputsFc1, 8)
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.unitRest1.forward(x)
        x = self.dropout(x)

        x = self.unitRest2.forward(x)
        x = self.dropout(x)

        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.avgPool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


net = Net()
print(net, flush=True)

# Convert Model to CUDA version
if ENABLE_GPU:
    net = net.cuda()

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0)
print(sum(p.numel() for p in net.parameters() if p.requires_grad))

# Training
if ENABLE_GPU:
    print("GPU loaded: " + str(torch.cuda.is_available()), flush=True)

print("Start training", flush=True)
correct = 0
total = 0
for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        if ENABLE_GPU:
            inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 25 == 24:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / batch_size), flush=True)
            #running_loss = 0.0
            #writer.add_scalar('Loss/train', running_loss/batch_size, epoch)

print('Accuracy training %d %%' % (100 * correct / total), flush=True)
print('Finished Training', flush=True)

# Train Model
print("Testing model ", flush=True)
correct = 0
total = 0
with torch.no_grad():
    for data in testLoader:
        images, labels = data

        if ENABLE_GPU:
            images, labels = images.cuda(), labels.cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy test %d %%' % (100 * correct / total), flush=True)
