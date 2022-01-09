import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils import data
import torch.nn as nn
import torch.functional as F
#Steps for creating a CNN
# 1.> Import data and create dataloader
# 2.> Create the CNN model
# 3.> Train the model
# 4.> Test model
# 5.> Optimize the model
# 6.> Capture image from camer and pass it to the model to find the result

# Hyperparameters declaration
epochs = 5
num_classes = 10
lr = 0.01
batch = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mnist_test = torchvision.datasets.MNIST(root="./data", download=True, transform=torchvision.transforms.ToTensor(), train=False)
mnist_train = torchvision.datasets.MNIST(root="./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)

test_data = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch, shuffle=True)
train_data = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch, shuffle=True)

class MnistCnn(nn.Module):
    def __init__(self):
        super(MnistCnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)


model = MnistCnn()
print(model)

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output
new_model = Net()
print(new_model)