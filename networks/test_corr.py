import torch
from correlation1d_package.modules.correlation1d import Correlation1D
from correlation_package.modules.correlation import Correlation
import torch.nn as nn
import math
import numpy as np
from submodules import *

# corr = nn.Sequential(
#             tofp32(),
#             Correlation(pad_size=1, kernel_size=1, max_displacement=1, stride1=1, stride2=1, corr_multiply=1),
#             tofp16())
corr1d = Correlation1D(pad_size=1, kernel_size=1, max_displacement=1, stride1=1, stride2=1, corr_multiply=1)
corr = Correlation(pad_size=2, kernel_size=1, max_displacement=2, stride1=1, stride2=1, corr_multiply=1)

x1 = torch.tensor(range(1,17),dtype=torch.float).view(1,1,4,-1).cuda()
x1 = torch.cat((x1,x1,x1),1)
x2 = torch.tensor(range(1,17),dtype=torch.float).view(1,1,4,-1).cuda()
x2 = torch.cat((x1,x1,x1),1)
print x1
result = corr1d(x1,x2)
save = True
res = torch.autograd.gradcheck(corr1d, (x1, x2), raise_exception=False)
print res
print result
