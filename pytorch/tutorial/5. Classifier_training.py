import torch
import torchvision
import torchvision.transforms as transforms

# torchvision 데이터셋의 출력은 [0, 1] 범위를 갖는 PILImage
# [-1, 1]의 범위로 정규화된 Tensor로 변환
# Windows 환경에서 BrokenPipeError 발생 시 DataLoader의 num_workers를 0으로 설정
# num_workers = 데이터 로드 멀티 프로세싱 관련

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 불러온 이미지 테스트 출력
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F


# 신경망 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # n,3,32,32 -> n,6,28,28 -> n,6,14,14 (n은 사진의 수)
        x = self.pool(F.relu(self.conv2(x))) # n,6,14,14 -> n,16,10,10 -> n,16,5,5
        x = x.view(-1, 16 * 5 * 5) # n,16x5x5 2차원 배열
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    # input -> Conv -> relu -> pool -> Conv -> relu -> pool -> fc1 -> relu -> fc2 -> relu -> fc3 -> output

net = Net()
# GPU 사용
device = torch.device("cuda:0")
net.to(device)

# 손실 함수, Optimizer 정의
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device) # trainloader 데이터 받아오기(x, y)

        optimizer.zero_grad() # 변화도를 0으로 만들어주고

        outputs = net(inputs) # 순전파
        loss = criterion(outputs, labels) # loss 계산
        loss.backward() # 역전파
        optimizer.step() # weight 최적화

        running_loss += loss.item()
        if i % 2000 == 1999: # 2000번의 mini-batch마다 출력
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished Training')

# 모델 저장
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# testdata를 사용한 평가
dataiter = iter(testloader)
images, labels = dataiter.next()[0].to(device), dataiter.next()[1].to(device)

imshow(torchvision.utils.make_grid(images.cpu())) # 불러온 이미지 출력
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4))) # 불러온 이미지가 무엇인지 출력

outputs = net(images) # testdata 이미지를 모델에 입력한 결과, 10개의 값을 갖는 배열, 각 class에 근접한 정도의 값

_, predicted = torch.max(outputs, 1) # 가장 높은 한개의 class, 해당 label만 사용

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4))) # 예측 결과

# 전체 testset에 대한 accuracy 확인
correct = 0
total = 0
with torch.no_grad(): # 학습이 아닌 평가이므로 grad 계산 불필요
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# 각 class별 예측 정확도 파악
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4): # 이미지 4개씩 불러옴
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))