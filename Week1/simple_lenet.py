import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

# Constants
ENABLE_GPU = True
LOCAL = True
GPU_ID = 0
if LOCAL:
    ENABLE_GPU = False
    DATASET_TRAIN = '/Users/kevinmartinfernandez/Workspace/Master/M3/BagOfWords/MIT_split/train/'
    DATASET_TEST = '/Users/kevinmartinfernandez/Workspace/Master/M3/BagOfWords/MIT_split/test/'
else:
    DATASET_TRAIN = '/home/mcv/datasets/MIT_split/train'
    DATASET_TEST = '/home/mcv/datasets/MIT_split/test'
EPOCHS = 100
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
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=(len(trainDataset)//batch_size), shuffle=True)

testDataset = datasets.ImageFolder(DATASET_TEST, transform=transform)
testLoader = torch.utils.data.DataLoader(testDataset, shuffle=True)


# Define Model
class LeNetSimple(nn.Module):
    def __init__(self):
        super(LeNetSimple, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.avgPool1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.avgPool2 = nn.AvgPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.avgPool3 = nn.AvgPool2d(2, stride=2)
        self.fc1 = nn.Linear(32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.avgPool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgPool2(x)
        x = F.relu(self.conv3(x))
        x = self.avgPool3(x)
        x = x.view(-1, 32*1*1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


net = LeNetSimple()
print(net, flush=True)
print("Parameters %d" % sum(p.numel() for p in net.parameters() if p.requires_grad))

# Convert Model to CUDA version
if ENABLE_GPU:
    net = net.cuda()

# Training
if ENABLE_GPU:
    print("GPU loaded: " + str(torch.cuda.is_available()), flush=True)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter("runs/LeNetSimpleLR0.001Batch200RMSProp")

print("Start training", flush=True)
# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.2)

correct = 0
total = 0

for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    epoch_loss = 0.0
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
        epoch_loss += loss.item()
        running_loss += loss.item()
        if i % 50 == 49:  # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50), flush=True)
            running_loss = 0.0

    writer.add_scalar('Training/Loss', epoch_loss / batch_size, epoch)
    writer.add_scalar('Training/Accuracy', (100 * correct / total), epoch)
    writer.flush()
    print("Epochs Loss: %.3f" % (epoch_loss / batch_size), flush=True)
    print("Epochs Acc: %.3f" % (100 * correct / total), flush=True)

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
