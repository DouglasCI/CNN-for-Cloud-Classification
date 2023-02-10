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

        # first set of Conv -> ReLU -> Max Pool layers
        self.conv1 = nn.Conv2d(in_channels=num_color_channels, out_channels=20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # second set of Conv -> ReLU -> Max Pool layers
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # FC -> ReLU -> FC layers
        self.fc1 = nn.Linear(in_features=8450, out_features=500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=500, out_features=num_classes)
        # Softmax classifier
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, h):
        h = self.conv1(h)
        # Conv1 -> ReLU1
        h = self.relu1(h)
        # ReLU1 -> maxpool1
        h = self.maxpool1(h)
        # maxpool1 -> Conv2
        h = self.conv2(h)
        # Conv2 -> ReLU2
        h = self.relu2(h)
        # ReLU2 -> maxpool2
        h = self.maxpool2(h)

        h = flatten(h, 1) #flatten

        h = self.fc1(h)
        # FC1 -> ReLU3
        h = self.relu3(h)
        # ReLU3 -> FC2
        h = self.fc2(h)
        # FC2 -> Softmax
        return self.logSoftmax(h)