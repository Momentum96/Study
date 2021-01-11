import torch

x = torch.ones(2, 2, requires_grad=True) # 연산을 기록
print(x)

y = x + 2 # 연산 수행
print(y)

print(y.grad_fn) # 연산 결과이므로 grad_fn을 가짐

z = y*y*3 # z = (x+2)^2*3
out = z.mean()

print(z, out)

"""
.requires_gard_(...)는 기존 Tensor의 requires_grad 값을 inplace 방식으로 변경. 입력값 지정되지 않으면 default는 False
"""

a = torch.randn(2, 2) # 기본 requires_gard = False
a = ((a * 3) / (a - 1))
print(a.requires_grad) # False
a.requires_grad_(True) # _가 뒤에 붙어있기 때문에 inplace 연산
print(a.requires_grad) # True
b = (a * a).sum() # True의 연산 결과이므로 grad_fn 가짐
print(b.grad_fn)
print(b)

# 역전파(backprop)

"""
1. scalar 값의 backprop
 = 미분 결과에 해당 값을 넣었을 때.

x = 2행 2열짜리 1이 들어있는 행렬
y = x+2 == 2행 2열짜리 3이 들어있는 행렬
z = y*y*3 == 2행 2열 27이 들어있는 행렬 (3*(x+2)^2)
out = z의 평균
"""
out.backward() # == out.backward(torch.tensor(1.))

print(out)
print(x.grad) # x가 사용된 최종 연산식에서 x에대한 미분 (out = 3*(x+2)^2/4 -> 3*(x+2)/2) (x = 1) (x = 1은 out.backward()에서 설정됨)

"""
2. vector의 backprop
 = Jacobian Matrix
 (m차원에서 n차원으로 가는 함수가 있다고 할 때 각각 차원에 대해 모든 편미분 값을 모아놓은 matrix)
 일반적인 미분 : 1변수 함수만 다룸
 편미분 : 다변수 함수에서 한 변수만 변수로, 나머지 변수는 상수로
 vector의 backprop은 편미분값 matrix에 곱해주는 어떤 값
"""

x = torch.rand(3, requires_grad=True)

y = x * 2 # 2x
while y.data.norm() < 1000:
    y = y * 2 # 2^n*x

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v) # 2^n에 0.1, 1.0, 0.0001 곱해준 값

print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad(): # Tensor 연산 기록 멈춤
    print((x ** 2).requires_grad)

print(x.requires_grad)
y = x.detach() # content는 같지만 require_grad가 False인 새로운 Tensor 생성
print(y.requires_grad)
print(x.eq(y).all())