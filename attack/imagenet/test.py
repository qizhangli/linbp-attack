import os, sys
import torch
import models as MODEL
import torchvision.transforms as T
import torchvision
import argparse
from torch.backends import cudnn
import numpy as np
import torch.nn.functional as F

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

def normalize(x, ms=None):
    if ms == None:
        ms = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - ms[0][i]) / ms[1][i]
    return x

def test(model, trans):
    target = torch.from_numpy(np.load(args.dir + '/labels.npy')).long()
    if 'target' in args.dir:
        label_switch = torch.tensor(list(range(500, 1000)) + list(range(0, 500))).long()
        target = label_switch[target]
    img_num = 0
    count = 0
    advfile_ls = os.listdir(args.dir)
    for advfile_ind in range(len(advfile_ls)-1):
        adv_batch = torch.from_numpy(np.load(args.dir + '/batch_{}.npy'.format(advfile_ind))).float() / 255
        if advfile_ind == 0:
            adv_batch_size = adv_batch.shape[0]
        img = adv_batch
        img_num += img.shape[0]
        label = target[advfile_ind * adv_batch_size : advfile_ind*adv_batch_size + adv_batch.shape[0]]
        label = label.to(device)
        img = img.to(device)
        with torch.no_grad():
            pred = torch.argmax(model(trans(img)), dim=1).view(1,-1)
        count += (label != pred.squeeze(0)).sum().item()
        del pred, img
        del adv_batch
    return round(100. - 100. * count / img_num, 2) if 'target' in args.dir else round(100. * count / img_num, 2)


inceptionv3 = MODEL.inceptionv3.Inception3()
inceptionv3.to(device)
inceptionv3.load_state_dict(torch.load('attack/imagenet/models/ckpt/inception_v3_google-1a9a5a14.pth'))
inceptionv3.eval()

def trans_incep(x):
    if 'incep' in args.dir:
        return normalize(x, ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]).data
    else:
        x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
        x = x[:, :, (256-224)//2: (256-224)//2 + 224, (256-224)//2: (256-224)//2 + 224]
        x = F.interpolate(x, size=(299,299))
        return normalize(x, ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]).data
print('inceptionv3:', test(model = inceptionv3, trans = trans_incep))
del inceptionv3

pnasnet = MODEL.pnasnet.pnasnet5large(ckpt_dir ='attack/imagenet/models/ckpt/pnasnet5large-bf079911.pth', num_classes=1000, pretrained='imagenet')
pnasnet.to(device)
pnasnet.eval()
def trans_pnas(x):
    x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
    x = x[:, :, (256-224)//2: (256-224)//2 + 224, (256-224)//2: (256-224)//2 + 224]
    x = F.interpolate(x, size=(331,331))
    return normalize(x, ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]).data
print('pnasnet:', test(model = pnasnet, trans = trans_pnas))
del pnasnet

senet = MODEL.senet.senet154(ckpt_dir ='attack/imagenet/models/ckpt/senet154-c7b49a05.pth')
senet.to(device)
senet.eval()
def trans_se(x):
    x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
    x = x[:, :, (256-224)//2: (256-224)//2 + 224, (256-224)//2: (256-224)//2 + 224]
    return normalize(x, ms = None).data
print('senet:', test(model = senet, trans = trans_se))
del senet

densenet = torchvision.models.densenet121(pretrained=False)
densenet.to(device)
import re
pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

state_dict = torch.load('attack/imagenet/models/ckpt/densenet121-a639ec97.pth')
for key in list(state_dict.keys()):
    res = pattern.match(key)
    if res:
        new_key = res.group(1) + res.group(2)
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
densenet.load_state_dict(state_dict)
densenet.eval()
print('densenet:', test(model = densenet, trans = trans_se))
del densenet

mobilenet = torchvision.models.mobilenet_v2(pretrained=False)
mobilenet.to(device)
mobilenet.load_state_dict(torch.load('attack/imagenet/models/ckpt/mobilenet_v2-b0353104.pth'))
mobilenet.eval()
print('mobilenet:', test(model = mobilenet, trans = trans_se))
del mobilenet

def trans_ori(x):
    if 'incep' in args.dir:
        x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
        x = x[:, :, (256-224)//2: (256-224)//2 + 224, (256-224)//2: (256-224)//2 + 224]
        return normalize(x, ms = None).data
    else:
        return normalize(x, ms = None).data
resnet50 = MODEL.resnet.resnet50(state_dict_dir ='attack/imagenet/models/ckpt/resnet50-19c8e357.pth')
resnet50.eval()
resnet50.to(device)
print('resnet50:', test(model = resnet50, trans = trans_ori))
del resnet50