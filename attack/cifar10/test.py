import os, sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import models
# import numpy as np
import torchvision.datasets as DATASETS
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dir', type=str, default='')
args = parser.parse_args()
print(args)

cudnn.benchmark = False
cudnn.deterministic = True
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

target = torch.from_numpy(np.load(args.dir + '/labels.npy')).long()
if 'target' in args.dir:
    label_switch = torch.tensor([1,2,3,4,5,6,7,8,9,0]).long()
    target = label_switch[target]


gdas = models.__dict__['gdas']('attack/cifar10/models/ckpt/gdas-cifar10-best.pth')
gdas.to(device)
gdas.eval()

pyramidnet = models.__dict__['pyramidnet272'](num_classes = 10)
pyramidnet.load_state_dict(torch.load('attack/cifar10/models/ckpt/pyramidnet272-checkpoint.pth', map_location=device)['state_dict'])
pyramidnet.to(device)
pyramidnet.eval()

ResNeXt_29_8_64d = models.__dict__['resnext'](
                cardinality=8,
                num_classes=10,
                depth=29,
                widen_factor=4,
                dropRate=0,
            )
ResNeXt_29_8_64d = nn.DataParallel(ResNeXt_29_8_64d)
ResNeXt_29_8_64d.load_state_dict(torch.load('attack/cifar10/models/ckpt/resnext-8x64d/model_best.pth.tar', map_location=device)['state_dict'])
ResNeXt_29_8_64d.eval()

DenseNet_BC_L190_k40 = models.__dict__['densenet'](
                num_classes=10,
                depth=190,
                growthRate=40,
                compressionRate=2,
                dropRate=0,
            )
DenseNet_BC_L190_k40 = nn.DataParallel(DenseNet_BC_L190_k40)
DenseNet_BC_L190_k40.load_state_dict(torch.load('attack/cifar10/models/ckpt/densenet-bc-L190-k40/model_best.pth.tar', map_location=device)['state_dict'])
DenseNet_BC_L190_k40.eval()

WRN = models.__dict__['wrn'](
                num_classes=10,
                depth=28,
                widen_factor=10,
                dropRate=0.3,
            )
WRN = nn.DataParallel(WRN)
WRN.load_state_dict(torch.load('attack/cifar10/models/ckpt/WRN-28-10-drop/model_best.pth.tar', map_location=device)['state_dict'])
WRN.eval()

vgg = models.__dict__['vgg19_bn'](num_classes=10)
vgg.features = nn.DataParallel(vgg.features)
vgg.load_state_dict(torch.load('attack/cifar10/models/ckpt/vgg19_bn/model_best.pth.tar', map_location=(device))['state_dict'])
vgg.to(device)
vgg.eval()

def get_pred(model, inputs):
    return torch.argmax(model(inputs), dim=1).view(1,-1)
def normal(x):
    ms = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
    for i in range(x.shape[1]):
        x[:,i,:,:] = (x[:,i,:,:] - ms[0][i]) / ms[1][i]
    return x

vgg_fool = 0
WRN_fool = 0
ResNeXt_29_8_64d_fool = 0
DenseNet_BC_L190_k40_fool = 0
pyramidnet_fool = 0
gdas_fool = 0
advfile_ls = os.listdir(args.dir)
img_num = 0
for advfile_ind in range(len(advfile_ls)-1):
    adv_batch = torch.from_numpy(np.load(args.dir + '/batch_{}.npy'.format(advfile_ind))).float() / 255
    if advfile_ind == 0:
        adv_batch_size = adv_batch.shape[0]

    inputs_ori = adv_batch
    img_num += inputs_ori.shape[0]
    labels = target[advfile_ind*adv_batch_size : advfile_ind*adv_batch_size + adv_batch.shape[0]]
    inputs = normal(inputs_ori.clone())
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        WRN_pred = get_pred(WRN, inputs)
        ResNeXt_29_8_64d_pred = get_pred(ResNeXt_29_8_64d, inputs)
        DenseNet_BC_L190_k40_pred = get_pred(DenseNet_BC_L190_k40, inputs)
        pyramidnet_pred = get_pred(pyramidnet, inputs)
        gdas_pred = get_pred(gdas, inputs)
        vgg_pred = get_pred(vgg, inputs)
    WRN_fool += (labels != WRN_pred.squeeze(0)).sum().item()
    ResNeXt_29_8_64d_fool += (labels != ResNeXt_29_8_64d_pred.squeeze(0)).sum().item()
    DenseNet_BC_L190_k40_fool += (labels != DenseNet_BC_L190_k40_pred.squeeze(0)).sum().item()
    pyramidnet_fool += (labels != pyramidnet_pred.squeeze(0)).sum().item()
    gdas_fool += (labels != gdas_pred.squeeze(0)).sum().item()
    vgg_fool += (labels != vgg_pred.squeeze(0)).sum().item()

def get_success_rate(fool_num, all_num):
    return 1 - fool_num / all_num if 'target' in args.dir else fool_num / all_num


print('vgg19_bn', get_success_rate(vgg_fool, img_num))
print('WRN', get_success_rate(WRN_fool, img_num))
print('ResNeXt_29_8_64d', get_success_rate(ResNeXt_29_8_64d_fool, img_num))
print('DenseNet_BC_L190_k40', get_success_rate(DenseNet_BC_L190_k40_fool, img_num))
print('pyramidnet', get_success_rate(pyramidnet_fool, img_num))
print('gdas', get_success_rate(gdas_fool, img_num))