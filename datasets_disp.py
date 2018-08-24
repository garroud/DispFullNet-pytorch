import torch
import torch.utils.data as data

import os, math, random
from os.path import *
import numpy as np

from glob import glob
import utils.frame_utils as frame_utils
import utils.readPFM as readPFM
from scipy.misc import imread, imresize

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)/2:(self.h+self.th)/2, (self.w-self.tw)/2:(self.w+self.tw)/2,:]

class FlyingThings(data.Dataset):
  def __init__(self, args, is_cropped, list_name='path/to/your/list', replicates = 1):
    self.args = args
    self.is_cropped = is_cropped
    self.crop_size = args.crop_size
    self.render_size = args.inference_size
    self.replicates = replicates
    self.data_path = list_name.split('/')[:-1]
    self.data_path = '/' + '/'.join(self.data_path) + '/'
    self.left_images = []
    self.right_images = []
    self.gt = []
    with open(list_name,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            self.left_images.append(line[0])
            self.right_images.append(line[1])
            self.gt.append(line[2])

    assert (len(self.left_images) == len(self.gt))
    assert (len(self.left_images) == len(self.right_images))
    self.size = len(self.left_images)

    self.frame_size = frame_utils.read_gen(self.data_path + self.left_images[0]).shape

    if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        self.render_size[0] = ( (self.frame_size[0])/64 ) * 64
        self.render_size[1] = ( (self.frame_size[1])/64 ) * 64

    args.inference_size = self.render_size

  def __getitem__(self, index):
    index = index % self.size

    img1 = frame_utils.read_gen(self.data_path + self.left_images[index])
    img2 = frame_utils.read_gen(self.data_path + self.right_images[index])

    flow, _ = readPFM.readPFM(self.data_path + self.gt[index])

    images = [img1, img2]
    image_size = img1.shape[:2]
    if self.is_cropped:
        cropper = StaticRandomCrop(image_size, self.crop_size)
    else:
        cropper = StaticCenterCrop(image_size, self.render_size)
    images = map(cropper, images)
    flow = cropper(flow)

    #This is for caffe reason?
    images = np.array(images).transpose(3,0,1,2)
    flow = flow.transpose(2,0,1)


    images = torch.from_numpy(images.astype(np.float32))
    flow = torch.from_numpy(flow.astype(np.float32))

    return [images], [flow]

  def __len__(self):
    return self.size * self.replicates

class FlyingThingsClean(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsClean, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_cleanpass', replicates = replicates)

class FlyingThingsFinal(FlyingThings):
    def __init__(self, args, is_cropped = False, root = '', replicates = 1):
        super(FlyingThingsFinal, self).__init__(args, is_cropped = is_cropped, root = root, dstype = 'frames_finalpass', replicates = replicates)


'''
import argparse
import sys, os
import importlib
from scipy.misc import imsave
import numpy as np

import datasets
reload(datasets)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.inference_size = [1080, 1920]
args.crop_size = [384, 512]
args.effective_batch_size = 1

index = 500
v_dataset = datasets.MpiSintelClean(args, True, root='../MPI-Sintel/flow/training')
a, b = v_dataset[index]
im1 = a[0].numpy()[:,0,:,:].transpose(1,2,0)
im2 = a[0].numpy()[:,1,:,:].transpose(1,2,0)
imsave('./img1.png', im1)
imsave('./img2.png', im2)
flow_utils.writeFlow('./flow.flo', b[0].numpy().transpose(1,2,0))

'''
