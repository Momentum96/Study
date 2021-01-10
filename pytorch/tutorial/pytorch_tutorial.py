from __future__ import print_function
import torch

x = torch.empty(5, 3) # 그 시점에 할당된 메모리에 존재하던 값들이 초기값으로 나타남
print(x)