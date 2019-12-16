import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


transform = transforms.Compose([transforms.Resize(32),
                                transforms.Scale((32,32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
                                ])

#the dataset used can be from https://www.kaggle.com/dansbecker/hot-dog-not-hot-dog/data
trainset = torchvision.datasets.ImageFolder(root='/content/gdrive/My Drive/train', transform=transform)

#checking the data
# x, y = trainset[50]
# x_pil = transforms.functional.to_pil_image(x).size

dataset_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)

def train():
    for epoch in range(30):
        print(epoch)
        for i, data in enumerate(dataset_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print("Finished training")

train()

def test():
    correct, total = 0, 0
    predictions = []
    net.eval()

    for i, data in enumerate(dataset_loader, 0):
        inputs, labels = data
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        predictions.append(outputs)
        total += labels.size(0)
        #print(labels)
        correct += (predicted == labels).sum().item()

    print("The testing set accuracy is {}".format(100 * correct/total))

test()

classes = ('hot dog', 'not hot dog')

# Using the neural net on specific images (not from training set)

from PIL import Image
from IPython.display import display
import torchvision.transforms.functional as TF

lst = [
       '/content/gdrive/My Drive/train/not_hot_dog/146029.jpg',
       '/content/gdrive/My Drive/download.jpg',
       '/content/gdrive/My Drive/burger.jpg',
       '/content/gdrive/My Drive/hd.jpg'
       ]

def eval_list(lst):

    imageloader = transforms.Compose([transforms.Resize(32),
                                    transforms.Scale((32,32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    net.eval()
    for x in lst:
        image = Image.open(x)
        display(image)  #if using an IPython notebook
        image = imageloader(image)
        image = image.unsqueeze(0)
        dogornot = net(image)
        _, predicted = torch.max(dogornot.data, 1)
        print(classes[predicted].upper())

eval_list()
