from time import time
import torch

A, B, C, D = 4, 512, 512, 512

cs = []
for i in range(10):
    c = torch.rand(A * B * C, D, dtype=torch.float16, device='cuda')
    cs.append(c)
    print(i + 1)


q = torch.rand(D, 1, dtype=torch.float16, device='cuda')

start_time = time()
ls = [c.matmul(q).squeeze(1).max() for c in cs]
# out = sorted(ls, key=lambda item: item[0])[0][1]
end_time = time()
print('%.2f' % (end_time - start_time))
