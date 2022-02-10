import PIL.Image
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as fn
import numpy
import torchvision
from torch.utils import data
from torchvision import transforms as T
from keras.preprocessing import image

# steps involved:
# 1. Download and define the training, testing data
# 2. Define the model by creating neural networks
# 3. Calculate the gradient, loss and update the model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transformation = torchvision.transforms.ToTensor()
transformation = T.Compose([
    T.ToTensor(),
    # T.Normalize((0.1307,), (0.3081,))
])

num_class = 10
# input_size = 784
# hidden_size = 100
batches = 100
learning_rate = 0.01
num_epochs = 4


mnist_train = torchvision.datasets.MNIST(root="./data", train=True, transform=transformation, download=True)
mnist_test = torchvision.datasets.MNIST(root="./data", train=False, transform=transformation, download=False)

train_data = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batches, shuffle=True)
test_data = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batches, shuffle=False)

examples = iter(train_data)
samples, label = examples.next()
print(samples.shape, label.shape)

test_examples = iter(test_data)
test_samples, test_label = test_examples.next()
print(test_samples.shape, test_label.shape)

# for i in range(6):
#     plt.subplot(3,3, i+1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()


# plt.imshow(samples[99][0], cmap='gray')
# plt.show()


# 2. Create the neural network layers
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        # nn.ReLU(),
        # nn.AvgPool2d(2, stride=2),  # (N, 3, 24, 24) -> (N,  3, 12, 12)
        # nn.Conv2d(3, 6, 3),
        # nn.BatchNorm2d(6)           # (N, 3, 12, 12) -> (N,  6, 10, 10)

    )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1,  padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(7*7*64, 1000)
        self.fc2 = nn.Linear(1000, 10)
    # self.features1 = nn.Sequential(
    #     nn.ReLU(),
    #     nn.AvgPool2d(2, stride=2)   # (N, 6, 10, 10) -> (N,  6, 5, 5)
    # )
    # self.classifier = nn.Sequential(
    #     nn.Linear(150, 25),         # (N, 150) -> (N, 25)
    #     nn.ReLU(),
    #     nn.Linear(25,10)            # (N, 25) -> (N, 10)
    # )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.fc2(out)
        # x = self.features1(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return out
    # def __init__(self, input_size, num_class):
    #     super(NeuralNetwork, self).__init__()
    #     self.conv1 = nn.Sequential(
    #         nn.Conv2d(
    #             in_channels=input_size, out_channels=32, kernel_size=(2,2), padding=2
    #         ),
    #         # nn.ReLU(),
    #         # nn.MaxPool2d(kernel_size=2),
    #     )
    #     self.maxp = nn.MaxPool2d(kernel_size=(2,2))
    #     self.conv2 = nn.Sequential(
    #         nn.Conv2d(
    #             in_channels=32,
    #             out_channels=64,
    #             kernel_size=(3,3),
    #             padding=0,
    #             stride=1
    #         ),
    #         # nn.ReLU(),
    #         # nn.MaxPool2d(2),
    #     )
    #     self.maxp2 = nn.MaxPool2d(kernel_size=(2,2))
    #     self.fc1 = nn.Sequential(
    #         nn.Linear(in_features=64*5*5, out_features=200)
    #     )
    #     self.fc2 = nn.Sequential(
    #         nn.Linear(in_features=200, out_features=num_class),
    #         nn.ReLU()
    #     )
    #     # fully connected layer, output 10 classes
    #     # self.out = nn.Linear(32 * 7 * 7, 10)
    #
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.maxp1(x)
    #     x = self.conv2(x)
    #     x = self.maxp2(x)
    #     x = x.contiguous().view(x.size(0), -1)
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     return x

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.conv2(x)
    #     # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
    #     x = x.view(x.size(0), -1)
    #     output = self.out(x)
    #     return output


model = NeuralNetwork()

# 3. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # 4. Training loop
n_steps = len(train_data)
for epoch in range(num_epochs):
    for i, datas in enumerate(samples):
# reshape the images
#         inputs, label = datas
#         inputs,labels = data[0].to(device), data[1].to(device)
#         inputs, label = datas[0].to(device), datas[1].to(device)
#         samples = np.expand_dims(datas, 1)    # if numpy array
#         images = torch.unsqueeze(datas, 1).to(device)
        # images = images.reshape(-1, 28*28).to(device)
        # label = label.to(device)
# forward pass
        outputs = model(samples )
        loss = criterion(outputs, label)

# backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1)%10 ==0 :
            print(f"epoch {epoch+1} loss = {loss.item()}")

# test and evaluate
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_data:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))



#
# cap = cv2.VideoCapture(0)
# while True:
#     status, photo = cap.read()
#     cv2.imshow('Attention please', photo)
#     if cv2.waitKey(1) == 13:
#         img = cv2.resize(photo, (28, 28))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.imwrite("new.png", photo)
#         break
# cv2.destroyAllWindows()

# photo = torch.from_numpy(photo)
# photo = photo.reshape(-1, 784).to(device)
# new = cv2.resize(photo, (28,28))

# crop = fn.center_crop(images, output_size=[28])
# resize = fn.resize(images, size=[28])

# new = preprocess(img)
# print(new.shape)


# img = img_transform(img)

#
# test_image = test_samples[74][0]
# test_image = test_image.reshape(-1, 784).to(device)
# images = torch.from_numpy(photo)
# # width, height = test_image.size
# crop = images.resize((784, int(784*(height/width))) if width < height else (int(784*(width/height)), 784))
# crop = fn.center_crop(images, output_size=[28])
# images = images.reshape(-1, 784).to(device)
# images = torchvision.transforms.CenterCrop(size=784)
# print(type(crop))
# print(crop.shape)
# print(model(test_image))
# plt.imshow(test_samples[74][0], cmap='gray')
# plt.show()
# output = model(img)
# print(output)

# newimg = cv2.imread("new.png")
# test_image = image.load_img("new.png", target_size = (28,28))
# test_image = test_samples[74][0]
# test_image = test_image.reshape(-1, 784).to(device)
# images = torch.from_numpy(test_image)
# width, height = test_image.size
# output = model(test_image)
# print(output)
# plt.imshow(newimg)
# plt.show()
