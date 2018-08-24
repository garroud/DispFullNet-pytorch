'''
Portions of this code copyright 2017, Clement Pinard
'''

# freda (todo) : adversarial loss

import torch
import torch.nn as nn
import math

#A layer to support nearest neighbor downsampling, currently only support one channel input with no backprop
#factor: resize the iamge to 1/factor with nearest neighbor model
class NN_downsample(nn.Module):
    def __init__(self,factor=2):
        super(NN_downsample,self).__init__()
        self.factor = factor
        self.kernel = torch.tensor((),dtype=torch.float, requires_grad=False).new_zeros((factor,factor))
        self.kernel[factor//2][factor//2] = 1
        self.kernel = self.kernel.view(1,1,factor,factor).cuda()
    def forward(self, x):
        return nn.functional.conv2d(x,self.kernel,stride=self.factor,padding=0)

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

#L1 Loss with mask
class L1Kitti(nn.Module):
    def __init__(self):
        super(L1Kitti,self).__init__()
    def forward(self,output, target):
        mask = (target != 0)
        mask.detach_()
        lossvalue = (output * mask - target) / mask.sum()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class MultiScale(nn.Module):
    def __init__(self, args, startScale = 1, numScales = 7, l_weight= 1, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        # self.loss_weights = torch.FloatTensor([max(l_weight - 0.2 * scale, 0.2) for scale in range(self.numScales)])
        self.loss_weights = torch.FloatTensor([1,0.8,0.6,0.4,0.2,0.2,0.03125]).cuda()
        self.args = args
        self.l_type = norm
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        elif self.l_type == 'L1Kitti':
            self.loss = L1Kitti()
        else:
            self.loss= L2()

        # self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.multiScales = [NN_downsample(self.startScale*(2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0
        if type(output) is tuple:
            print 'ok'
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                lossvalue += self.loss_weights[i] * self.loss(output_, target_)
                if i == 0:
                    epevalue = self.loss_weights[i] * self.loss(output_, target_)
            return [lossvalue, epevalue]
        else:
            lossvalue += self.loss(output, target)
            epevalue = lossvalue
            return  [lossvalue, epevalue]
