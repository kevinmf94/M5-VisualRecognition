import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

# Constants
ENABLE_GPU = True
GPU_ID = 0
DATASET_TRAIN = '/home/mcv/datasets/MIT_split/train'
DATASET_TEST = '/home/mcv/datasets/MIT_split/test'
EPOCHS = 300
batch_size = 300
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

validationDataset = datasets.ImageFolder(DATASET_TEST, transform=transform)
validationLoader = torch.utils.data.DataLoader(validationDataset,  batch_size=int(len(trainDataset) / batch_size), shuffle=True)

testDataset = datasets.ImageFolder(DATASET_TEST, transform=transform)
testLoader = torch.utils.data.DataLoader(testDataset, shuffle=True)


# Define Model
class UnitRestNet(nn.Module):

    def __init__(self, input_dim, filters, kernel=3):
        super(UnitRestNet, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, filters, kernel, 1, 1)
        nn.init.xavier_uniform(self.conv1.weight)
        self.batchNorm1 = nn.BatchNorm2d(filters, eps=1e-3, momentum=0.99)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(filters, filters, kernel, 1, 1)
        nn.init.xavier_uniform(self.conv3.weight)
        self.batchNorm2 = nn.BatchNorm2d(filters, eps=1e-3, momentum=0.99)
        self.relu2 = nn.ReLU()
        self.conv4 = nn.Conv2d(filters, filters, kernel, 1, 1)
        nn.init.xavier_uniform(self.conv4.weight)

    def forward(self, x):
        x = self.conv1(x)
        res = x
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batchNorm2(x)
        x = self.relu2(x)
        x = self.conv4(x)
        x += res
        return x


class ResidualNetworkModify(nn.Module):
    def __init__(self):
        super(ResidualNetworkModify, self).__init__()

        self.dropoutValue = 0.4

        self.unitRest1 = UnitRestNet(3, 8, 3)
        self.dropout1 = nn.Dropout(self.dropoutValue)
        self.unitRest2 = UnitRestNet(8, 10, 3)
        self.dropout2 = nn.Dropout(self.dropoutValue)
        self.batchNorm1 = nn.BatchNorm2d(10, eps=1e-3, momentum=0.99)
        self.relu1 = nn.ReLU()
        self.dropout3 = nn.Dropout(self.dropoutValue)
        self.avgPool = nn.AvgPool2d(4, stride=None)
        self.inputsFc1 = 10 * (img_width // 4) * (img_height // 4)
        self.flatten = nn.Flatten()
        self.dropout4 = nn.Dropout(self.dropoutValue)
        self.fc1 = nn.Linear(self.inputsFc1, 8)
        nn.init.xavier_uniform(self.fc1.weight)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.unitRest1.forward(x)
        x = self.dropout1(x)
        x = self.unitRest2.forward(x)
        x = self.dropout2(x)
        x = self.batchNorm1(x)
        x = self.relu1(x)
        x = self.dropout3(x)
        x = self.avgPool(x)
        x = self.flatten(x)
        x = self.dropout4(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


net = ResidualNetworkModify()
writer = SummaryWriter("runs/rsesidualNetworkRmspropLR0_001M0_2_BATCH300_EP300DROP0_4_TEST_DROPOUT")
writer.add_text("Model/structure", str(net))
writer.add_text("Model/paramaters", "Parameters %d" % sum(p.numel() for p in net.parameters() if p.requires_grad))

# Convert Model to CUDA version
if ENABLE_GPU:
    net = net.cuda()

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.001, momentum=0.2)

# Training
if ENABLE_GPU:
    print("GPU loaded: " + str(torch.cuda.is_available()), flush=True)

print("Start traininig", flush=True)
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
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #Accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # print statistics
        running_loss += loss.item()
        epoch_loss += loss.item()
        if i % 50 == 49:  # print every 2000 mini-batches
            lossLog = '[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50)
            writer.add_text("Training/Losses", lossLog)
            print(lossLog, flush=True)
            running_loss = 0.0

    #Validation Test
    totalVal = 0.0
    correctVal = 0.0
    with torch.no_grad():
        dataiter = iter(validationLoader)
        images, labels = dataiter.next()

        if ENABLE_GPU:
            images, labels = images.cuda(), labels.cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        totalVal += labels.size(0)
        correctVal += (predicted == labels).sum().item()

    print("Epochs Loss: %.3f" % (epoch_loss / batch_size), flush=True)
    print("Epochs Val Acc: %.3f" % (100 * correctVal / totalVal), flush=True)
    print("Epochs Acc: %.3f" % (100 * correct / total), flush=True)
    writer.add_scalar('Validation/Accuracy', (100 * correct / total), epoch)
    writer.add_scalar('Training/Loss', epoch_loss / batch_size, epoch)
    writer.add_scalar('Training/Accuracy', (100 * correct / total), epoch)
    writer.flush()


print('Finished Training', flush=True)
accText = 'Accuracy training %d %%' % (100 * correct / total)
writer.add_text("Training/FinalAcc", accText)
print(accText, flush=True)
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

accText = 'Accuracy %d %%' % (100 * correct / total)
print(accText, flush=True)
writer.add_text("Test/Accuracy", accText)
writer.flush()

