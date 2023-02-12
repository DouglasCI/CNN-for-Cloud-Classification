from torch import flatten
from torch import nn


class CNN(nn.Module):
    """
    Class for a CNN model.
    """
    def __init__(self,
            num_classes=11,
            num_color_channels=3,
            ):

        super(CNN, self).__init__()
        
        # # first set of Conv -> ReLU -> Max Pool layers
        # self.conv1 = nn.Conv2d(in_channels=num_color_channels, out_channels=30, kernel_size=(5, 5))
        # self.relu1 = nn.ReLU()
        # self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        # # second set of Conv -> ReLU -> Max Pool layers
        # self.conv2 = nn.Conv2d(in_channels=30, out_channels=100, kernel_size=(5, 5))
        # self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        # # FC -> ReLU -> FC layers
        # self.fc1 = nn.Linear(in_features=44100, out_features=500)
        # self.relu3 = nn.ReLU()
        # self.fc2 = nn.Linear(in_features=500, out_features=num_classes)
        # # Softmax classifier
        # self.logSoftmax = nn.LogSoftmax(dim=1)
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_color_channels, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        # x = self.conv1(x)
        # # Conv1 -> ReLU1
        # x = self.relu1(x)
        # # ReLU1 -> maxpool1
        # x = self.maxpool1(x)
        # # maxpool1 -> Conv2
        # x = self.conv2(x)
        # # Conv2 -> ReLU2
        # x = self.relu2(x)
        # # ReLU2 -> maxpool2
        # x = self.maxpool2(x)
        # x = flatten(x, 1) #flatten
        # x = self.fc1(x)
        # # FC1 -> ReLU3
        # x = self.relu3(x)
        # # ReLU3 -> FC2
        # x = self.fc2(x)
        # # FC2 -> Softmax
        # return self.logSoftmax(x)
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out