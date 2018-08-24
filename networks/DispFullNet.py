import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from correlation1d_package.modules.correlation1d import Correlation1D

from submodules import *
'Parameter count , 39,175,298'

class DispFullNet(nn.Module):
    def __init__(self,args, batchNorm=False):
        super(DispFullNet,self).__init__()
        self.training = True
        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   3,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv_redir  = conv(self.batchNorm, 128, 64, kernel_size=1, stride=1)

        if args.fp16:
            self.corr = nn.Sequential(
                tofp32(),
                Correlation1D(pad_size=40, kernel_size=1, max_displacement=40, stride1=1, stride2=1, corr_multiply=1),
                tofp16())
        else:
            self.corr = Correlation1D(pad_size=40, kernel_size=1, max_displacement=40, stride1=1, stride2=1, corr_multiply=1)

        # self.corr_activation = nn.LeakyReLU(0.1,inplace=True)
        self.conv_2_reshape = conv(self.batchNorm, 145,  128)

        #128+81 = 209
        self.conv3 = conv(self.batchNorm, 128, 256, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(512,512)
        self.deconv3 = deconv(512,256)
        self.deconv2 = deconv(256,128)
        self.deconv1 = deconv(128,64)
        self.deconv0 = deconv(64,48)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(512)
        self.predict_flow3 = predict_flow(256)
        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(64)
        self.predict_flow0 = predict_flow(48)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

        self.Convolution5 = conv(self.batchNorm, 1025, 512)
        self.Convolution4 = conv(self.batchNorm, 1025, 512)
        self.Convolution3 = conv(self.batchNorm, 513, 256)
        self.Convolution2 = conv(self.batchNorm, 257, 128)
        self.Convolution1 = conv(self.batchNorm, 129, 64)
        self.Convolution0 = conv(self.batchNorm, 49, 48)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)
        # self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x1 = x[:,0:3,:,:]
        x2 = x[:,3::,:,:]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)

        # FlownetC bottom input stream
        out_conv1b = self.conv1(x2)

        out_conv2b = self.conv2(out_conv1b)

        # Merge streams
        out_corr = self.corr(out_conv2a, out_conv2b) # False
        # out_corr = self.corr_activation(out_corr)

        # Redirect top input stream and concatenate
        out_conv_redir = self.conv_redir(out_conv2a)

        concat_conv2 = torch.cat((out_conv_redir, out_corr), 1)
        out_conv2   = self.conv_2_reshape(concat_conv2)

        # Merged conv layers
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        convolution5 = self.Convolution5(concat5)

        flow5       = self.predict_flow5(convolution5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(convolution5)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        convolution4 = self.Convolution4(concat4)

        flow4       = self.predict_flow4(convolution4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(convolution4)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        convolution3 = self.Convolution3(concat3)

        flow3       = self.predict_flow3(convolution3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(convolution3)

        concat2 = torch.cat((out_conv2a,out_deconv2,flow3_up),1)
        convolution2 = self.Convolution2(concat2)

        flow2       = self.predict_flow2(convolution2)
        flow2_up    = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(convolution2)

        concat1 = torch.cat((out_conv1a,out_deconv1,flow2_up),1)
        convolution1 = self.Convolution1(concat1)

        flow1       = self.predict_flow1(convolution1)
        flow1_up    = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(convolution1)

        concat0 = torch.cat((out_deconv0,flow1_up),1)
        convolution0 = self.Convolution0(concat0)

        flow0 = self.predict_flow0(convolution0)

        if self.training:
            return nn.functional.relu(flow0),nn.functional.relu(flow1),nn.functional.relu(flow2), \
                   nn.functional.relu(flow3),nn.functional.relu(flow4),nn.functional.relu(flow5), \
                   nn.functional.relu(flow6)
        else:
            return flow0
