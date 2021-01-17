import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 입력 이미지 채널 1개, 출력 채널 6개, 3x3의 정사각 컨볼루션 행렬
        # 컨볼루션 커널 정의
        # troch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # in_channels(int) : input image의 channel수. rgb면 3
        # out_channels(int) : convolution에 의해 생성된 channel 수
        # kernel_size(int or tuple) : convoling_kernel 크기. (filter)
        # stride(int or tuple) : convolution의 stride를 얼만큼 줄 것인가. default는 1, stride는 이미지 횡단 시 커널의 스텝 사이즈
        # padding(int or tuple) : zero padding을 input 양쪽 인자만큼. default는 0이라서 기본적으로 설정하지 않을 경우 zero padding 적용하지 않음
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 아핀(affine) 연산: y = Wx + b
        self.fc1 = nn.Linear(16*6*6, 120) # 6*6은 이미지 차원
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # (2, 2) 크기 윈도우에 대해 맥스 풀링(max pooling)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 1,32,32 -> 6,30,30(Conv2d) -> 6,15,15(pool2d)
        # 크기가 제곱수라면 하나의 숫자만을 특정
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # 6,15,15 -> 16,13,13(Conv2d) -> 16,6,6(pool2d) (소수 버림)
        x = x.view(-1, self.num_flat_features(x)) # 1차원으로 변경 16x6x6개의 값
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # 배치 차원을 제외한 모든 차원
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# 모델의 학습 가능한 매개변수 확인
params = list(net.parameters())
print(len(params))
print(params[0].size())

# 임의의 32x32값 입력
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# 모든 매개변수의 변화도 버퍼를 0으로 설정, 무작위 값으로 역전파
net.zero_grad()
out.backward(torch.rand(1, 10))

# 손실 함수
# output, target을 한 쌍의 입력으로 받아, 출력이 정답으로부터 얼마나 멀리 떨어져있는지 추정하는 값을 계산
# 간단한 손실 함수로 출력과 대상 간 평균제곱오차를 계산하는 nn.MSEloss가 있음
output = net(input)
target = torch.randn(10) # 비교를 위한 예시, 임의의 정답
target = target.view(1, -1) # 출력과 같은 shape로 변환
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

"""
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
"""
# 변화도가 누적된 .grad Tensor
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

# 역전파
# 기존 변화도를 없애지 않으면 기존의 변화도에 누적됨
net.zero_grad()

print('conv1.bias.grad before backward') # 0으로 초기화되어있음
print(net.conv1.bias.grad)

loss.backward() # loss = criterion(output, target)

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 가중치 갱신
# 확률적 경사하강법
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# torch.optim이라는 작은 패키지에 SGD, Nesterov-SGD, Adam, RMSProp등 다양한 갱신 규칙을 구현해두었음
import torch.optim as optim

# Optimizer 생성
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 학습 과정
optimizer.zero_grad() # optimizer.zero_grad()를 사용하여 수동으로 변화도 버퍼를 0으로 설정
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()