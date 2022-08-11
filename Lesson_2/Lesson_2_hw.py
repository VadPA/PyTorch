import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from torch import optim
from torch import nn
# from tensorflow import keras
# from keras.datasets import cifar10

import matplotlib.pyplot as plt

import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16 * hidden_dim)
        self.fc2 = nn.Linear(16 * hidden_dim, 16 * hidden_dim)
        self.fc3 = nn.Linear(16 * hidden_dim, 8 * hidden_dim)
        self.fc4 = nn.Linear(8 * hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        x = F.leaky_relu(x)
        x = self.fc5(x)
        return x

    def predict(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        x = F.leaky_relu(x)
        x = self.fc5(x)
        x = F.softmax(x)
        return x


train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)

# image, label = train_dataset[0]
# print(image.size())
# print(label)
#
# print(image.permute(1, 2, 0).shape)
#
# plt.imshow(image.permute(1, 2, 0).numpy())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8,
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net(3072, 32, 10)
print(net.train())

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001055, weight_decay=1e-6)  # .SGD  momentum=0.0

num_epochs = 45

for epoch in range(num_epochs):
    running_loss = 0.0
    running_items = 0.0


    for i, data in enumerate(train_loader):
        inputs, labels = data[0], data[1]

         # Обнуляем градиент
        optimizer.zero_grad()
        # Делаем предсказание
        outputs = net(inputs)
        # Рассчитываем лосс-функцию
        loss = criterion(outputs, labels)
        # Делаем шаг назад по лоссу
        loss.backward()
        # Делаем шаг нашего оптимайзера
        optimizer.step()

        # выводим статистику о процессе обучения
        running_loss += loss.item()
        running_items += len(labels)
        if i % 300 == 0:    # печатаем каждые 300 mini-batches
            print(f'Epoch [{epoch + 1}/{num_epochs}]. '
                  f'Step [{i + 1}/{len(train_loader)}]. '
                  f'Loss: {running_loss / running_items:.3f}')
            running_loss, running_items = 0.0, 0.0

print('Training is finished!')

# Загрузка и сохранение модели

# Сохранение модели
PATH_WEIGHTS = './cifar_net_test_version_21_weights.pth'
torch.save(net.state_dict(), PATH_WEIGHTS)

print("Model state_dict: ")

for param in net.state_dict():
    print(param, "\t", net.state_dict()[param].size())

PATH_MODEL = './cifar_net_test_version_21_model.pth'
torch.save(net, PATH_MODEL)

### Загрузка и использование модели

# net = Net(3072, 32, 10)
# net.load_state_dict(torch.load(PATH_WEIGHTS))
# print(net)
#
# net = torch.load(PATH_MODEL)

data_iter = iter(test_loader)
images, labels = data_iter.next()

net.eval()
outputs = net(images)
imgs = torchvision.utils.make_grid(images)
plt.figure(figsize=(10, 5))
plt.imshow(imgs.permute(1, 2, 0).numpy())
print('GroundTruth: ', ' '.join(classes[labels[j]] for j in range(len(labels))))

print(outputs)

net.predict(images)

_, predicted = torch.max(outputs, 1)

print(predicted)

print('Predicted: ', ' '.join(classes[predicted[j]] for j in range(len(labels))))

gt = np.array([classes[labels[j]] for j in range(len(labels))])
pred = np.array([classes[predicted[j]] for j in range(len(labels))])

print(gt)
print(pred)
print(f'Accuracy is {(gt == pred).sum() / len(gt)}')

print()
