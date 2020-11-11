import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch.nn as nn
import argparse
import models
import torch.nn.functional as F
from torch.backends import cudnn
import pickle
import numpy as np
import csv
import PIL.Image as Image

# Selected cifar-10. The .csv file format:
# class_index,data_index
# 3,0
# 8,1
# 8,2
# ...
class SelectedCifar10(Dataset):
    def __init__(self, cifar10_dir, selected_images_csv, transform=None):
        super(SelectedCifar10, self).__init__()
        self.cifar10_dir = cifar10_dir
        self.data = []
        self.targets = []
        file_path = os.path.join(cifar10_dir, 'test_batch')
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.transform = transform
        self.selected_images_csv = selected_images_csv
        self._load_csv()
    def _load_csv(self):
        reader = csv.reader(open(self.selected_images_csv, 'r'))
        next(reader)
        self.selected_list = list(reader)
    def __getitem__(self, item):
        t_class, t_ind = map(int, self.selected_list[item])
        assert self.targets[t_ind] == t_class, 'Wrong targets in csv file.(line {})'.format(item+1)
        img, target = self.data[int(self.selected_list[item][1])], self.targets[t_ind]
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    def __len__(self):
        return len(self.selected_list)

class Normalize(nn.Module):
    def __init__(self,):
        super(Normalize, self).__init__()
        self.ms = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
    def forward(self, input):
        x = input.clone()
        for i in range(x.shape[1]):
            x[:,i] = (x[:,i] - self.ms[0][i]) / self.ms[1][i]
        return x

# input diversity for mdi2fgsm
def input_diversity(img):
    gg = torch.randint(0, 2, (1,)).item()
    if gg == 0:
        return img
    else:
        rnd = torch.randint(32,41, (1,)).item()
        rescaled = F.interpolate(img, (rnd, rnd), mode = 'nearest')
        h_rem = 40 - rnd
        w_hem = 40 - rnd
        pad_top = torch.randint(0, h_rem + 1, (1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_hem + 1, (1,)).item()
        pad_right = w_hem - pad_left
        padded = F.pad(rescaled, pad = (pad_left, pad_right, pad_top, pad_bottom))
        padded = F.interpolate(padded, (32, 32), mode = 'nearest')
    return padded

# vgg-19 forward
def vgg19_forw(model, x, linbp, linbp_layer):
    x = model[0](x)
    for ind, mm in enumerate(model[1].features.module):
        if linbp and isinstance(mm, nn.ReLU) and ind >= linbp_layer:
            x = linbp_relu(x)
        else:
            x = mm(x)
    x = x.view(x.size(0), -1)
    x = model[1].classifier(x)
    return x

def linbp_relu(x):
    x_p = F.relu(-x)
    x = x + x_p.data
    return x

def vgg19_ila_forw(model, x, ila_layer):
    x = model[0](x)
    for ind, mm in enumerate(model[1].features.module):
        x = mm(x)
        if ind == ila_layer:
            return x

class ILAProjLoss(torch.nn.Module):
    def __init__(self):
        super(ILAProjLoss, self).__init__()
    def forward(self, old_attack_mid, new_mid, original_mid, coeff):
        n = old_attack_mid.shape[0]
        x = (old_attack_mid - original_mid).view(n, -1)
        y = (new_mid - original_mid).view(n, -1)
        # x_norm = x / torch.norm(x, dim = 1, keepdim = True)
        proj_loss = torch.sum(y * x) / n
        return proj_loss


