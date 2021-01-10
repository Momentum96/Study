import torch

x = torch.ones(2, 2, requires_grad=True) # 연산을 기록
print(x)

y = x + 2 # 연산 수행
print(y)

print(y.grad_fn) # 연산 결과이므로 grad_fn을 가짐

z = y*y*3
out = z.mean()

print(z, out)

"""
.requires_gard(...)는 기존 Tensor의 requires_grad 값을 inplace 방식으로 변경. 입력값 지정되지 않으면 default는 False
"""

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)