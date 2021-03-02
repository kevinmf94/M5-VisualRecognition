import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

# Constants
ENABLE_GPU = True
# DATASET_TRAIN = '/Users/kevinmartinfernandez/Workspace/Master/M3/BagOfWords/MIT_split/train/'
# DATASET_TEST = '/Users/kevinmartinfernandez/Workspace/Master/M3/BagOfWords/MIT_split/test/'
DATASET_TRAIN = '/home/mcv/datasets/MIT_split/train'
DATASET_TEST = '/home/mcv/datasets/MIT_split/test'
EPOCHS = 5
batch_size = 100
img_width = 128
img_height = 128

# Set GPU
if ENABLE_GPU:
    torch.cuda.set_device(device=0)

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

    def __init__(self, input_dim, filters, kernel=3, pool=False):
        super(UnitRestNet, self).__init__()

        self.isPool = pool

        self.conv1 = nn.Conv2d(input_dim, filters, kernel, 1, 1)
        self.maxPool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(filters, filters, 1, 2, 0)
        self.batchNorm1 = nn.BatchNorm2d(filters)
        self.conv3 = nn.Conv2d(filters, filters, kernel, 1, 1)
        self.batchNorm2 = nn.BatchNorm2d(filters)
        self.conv4 = nn.Conv2d(filters, filters, kernel, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        res = x

        if self.isPool:
            x = self.maxPool(x)
            res = self.conv2(res)

        x = F.relu(self.batchNorm1(x))
        x = self.conv3(x)
        x = F.relu(self.batchNorm2(x))
        x = self.conv4(x)
        x = torch.add(res, x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.unitRest1 = UnitRestNet(3, 8, 3)
        print(self.unitRest1)
        self.unitRest2 = UnitRestNet(8, 10, 3)
        self.avgPool = nn.AvgPool2d(2)
        self.inputsFc1 = 10 * 64 * 64
        self.fc1 = nn.Linear(self.inputsFc1, 8)
        self.softmax = nn.Softmax(dim=8)

    def forward(self, x):
        x = self.unitRest1.forward(x)
        x = F.dropout2d(x, 0.5)

        x = self.unitRest2.forward(x)
        x = F.dropout2d(x, 0.5)

        x = self.avgPool(x)

        x = x.view(-1, self.inputsFc1)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


net = Net()
print(net)

# Convert Model to CUDA version
if ENABLE_GPU:
    net = net.cuda()

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training
if ENABLE_GPU:
    print("GPU loaded: " + str(torch.cuda.is_available()))

for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 25 == 24:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / batch_size))
            running_loss = 0.0

print('Finished Training')

# Train Model
print("Testing model ")
correct = 0
total = 0
with torch.no_grad():
    for data in testLoader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy %d %%' % (100 * correct / total))
