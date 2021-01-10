from __future__ import print_function
import torch

x = torch.empty(5, 3) # 그 시점에 할당된 메모리에 존재하던 값들이 초기값으로 나타남
print(x)

x = torch.rand(5, 3) # 무작위로 초기화된 행렬 (0 <= x < 1)
print(x)

x = torch.zeros(5, 3, dtype=torch.long) # dtype = long, 0으로 채워진 행렬
print(x)

x = torch.tensor([5.5, 3]) # list를 사용하여 직접 tensor 생성
print(x)

x = x.new_ones(5, 3, dtype=torch.double) # 기존 tensor를 바탕으로 새로운 tensor 생성
print(x)

x = torch.randn_like(x, dtype=torch.float) # 기존 tensor를 바탕으로 새로운 tensor 생성
print(x)

print(x.size()) # 행렬 크기 구하기, 반환인 torch.Size는 튜플 타입, 모든 튜플 연산 지원

y = torch.rand(5, 3) # 연산 문법 1
print(x + y)

print(torch.add(x, y)) # 연산 문법 2

result = torch.empty(5, 3)
torch.add(x, y, out=result) # 연산 문법 3
print(result)

"""
    inplace 방식의 tensor값 변경하는 연산 뒤에는 _가 붇는다
    ex) x.copy_(y), x.t_()는 x 자체를 변경
"""

print(x[:, 1]) # 인덱싱 표기법

# tensor의 size, shape 변경 시에는 torch.view 사용
x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item()) # tensor에 하나의 값만 존재한다면 .item()을 사용하여 숫자 값 얻을 수 있음