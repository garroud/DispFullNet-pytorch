from torch.nn.modules.module import Module

from ..functions.correlation1d import Correlation1DFunction


class Correlation1D(Module):

    def __init__(self,
                 pad_size=1,
                 kernel_size=1,
                 max_displacement=1,
                 stride1=1,
                 stride2=1,
                 corr_multiply=1):
        super(Correlation1D, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self,input1, input2):
        return Correlation1DFunction.apply(input1, input2, self.pad_size,
                                         self.kernel_size,
                                         self.max_displacement, self.stride1,
                                         self.stride2, self.corr_multiply)
