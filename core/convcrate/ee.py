
import torch

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, mode='sintel', padding_factor=8):
        self.padding_factor = padding_factor
    def xx(self, ht):
        pad_ht = (((ht // self.padding_factor) + 1) * self.padding_factor - ht) % self.padding_factor
        pad_h = self.padding_factor - ht % self.padding_factor
        return pad_ht, pad_h, ht // self.padding_factor
    
# pad = InputPadder()
# for h in range(0, 1000):
#     print(h, pad.xx(h), "\n")

# a = [1, 2, 3, 4, 5]
# b = a[4:1:-1]
# print(b)


# y, x = torch.meshgrid(torch.arange(100), torch.arange(64))
# print(x,y,x.size(),y.size())
# a = torch.randn(2, 3, 100, 64)
# a = a.view(2,3,-1).permute(0,2,1)
# b = torch.randn(2, 3, 100, 64)
# b = b.view(2,3,-1)
# c = torch.matmul(a ,b).view(2,100,64,100,64)/(3 ** 0.5)
# print(c, c.shape)

# import numpy as np
# print((torch.ones([]) * np.log(1 / 0.07)).exp())

# import timm

# model = timm.list_models()
# print(model[1], model[1].split('_')[2])

import numpy as np
a = np.random.randn(100, 100)
print(a[:30,:30].copy().shape)