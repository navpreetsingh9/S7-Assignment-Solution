import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.05

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(64, 128, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(128, 256, 3) # 5 > 3 | 32 | 3*3*1024 | 3x3x1024x10 | 
        self.conv7 = nn.Conv2d(256, 10, 3) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), # 28>28 | 1>3 | 1>1
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.Conv2d(8, 8, 3, padding=1), # 28>28 | 3>5 | 1>1
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.Conv2d(8, 8, 3, padding=1), # 28>28 | 5>7 | 1>1
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2, 2)  # 28>14 | 7>8 | 1>2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 13, 3, padding=1), # 14>14 | 8>12 | 2>2
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value),
            nn.Conv2d(13, 13, 3, padding=1), # 14>14 | 12>16 | 2>2
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value),
            nn.Conv2d(13, 13, 3, padding=1), # 14>14 | 16>20 | 2>2
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2, 2) # 14>7 | 20>22 | 2>4
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(13, 18, 3), # 7>5 | 22>30 | 4>4
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout(dropout_value),
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(3), # 5>1 | 30>38 | 4>12
            nn.Conv2d(18, 10, 1) # 1>1 | 38>38 | 12>12
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(x, dim=1)
        return x        


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 7, 3, padding=1), # 28>28 | 1>3 | 1>1
            nn.ReLU(),
            nn.BatchNorm2d(7),
            nn.Dropout(dropout_value),
            nn.Conv2d(7, 7, 3, padding=1), # 28>28 | 3>5 | 1>1
            nn.ReLU(),
            nn.BatchNorm2d(7),
            nn.Dropout(dropout_value),
            nn.Conv2d(7, 7, 3, padding=1), # 28>28 | 5>7 | 1>1
            nn.ReLU(),
            nn.BatchNorm2d(7),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2, 2)  # 28>14 | 7>8 | 1>2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(7, 10, 3, padding=1), # 14>14 | 8>12 | 2>2
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.Conv2d(10, 10, 3, padding=1), # 14>14 | 12>16 | 2>2
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.Conv2d(10, 10, 3, padding=1), # 14>14 | 16>20 | 2>2
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2, 2) # 14>7 | 20>22 | 2>4
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 13, 3, padding=1), # 7>7 | 22>30 | 4>4
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value),
            nn.Conv2d(13, 13, 3, padding=1), # 7>7 | 30>38 | 4>4
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value),
            nn.MaxPool2d(2, 2), # 7>3 | 38>42 | 4>8
            nn.Conv2d(13, 13, 3, padding=1), # 3>3 | 42>58 | 8>8
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(dropout_value),
        )
        self.gap = nn.Sequential(
            nn.AvgPool2d(3), # 3>1 | 58>74 | 8>24
            nn.Conv2d(13, 10, 1) # 1>1 | 74>74 | 24>24
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(x, dim=1)
        return x            