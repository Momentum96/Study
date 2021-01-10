import torch
import numpy as np

x = torch.randn(1)

if torch.cuda.is_available(): # CUDA 사용 가능 환경(GPU 환경)에서만 실행
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device) # GPU 상 직접 tensor 생성
    x = x.to(device) # CPU 텐서를 GPU 텐서로 변경
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))