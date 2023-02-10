import os
import torch
import torchvision.transforms as T
import torch.nn as nn
import argparse
import models
from torch.backends import cudnn
import numpy as np
from utils import Normalize, input_diversity, vgg19_forw, vgg19_ila_forw, ILAProjLoss, SelectedCifar10

parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default = 0.03)
parser.add_argument('--niters', type=int, default = 100)
parser.add_argument('--ila_niters', type=int, default = 100)
parser.add_argument('--method', type=str, default = 'linbp_ila_pgd')
parser.add_argument('--linbp_layer', type=int, default = 23)
parser.add_argument('--ila_layer', type=int, default = 23)
parser.add_argument('--save_dir', type=str, default = '')
parser.add_argument('--batch_size', type=int, default = 500)
parser.add_argument('--target_attack', default=False, action='store_true')
args = parser.parse_args()



if __name__ == '__main__':
    print(args)
    cudnn.benchmark = False
    cudnn.deterministic = True
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


    os.makedirs(args.save_dir, exist_ok=True)
    epsilon = args.epsilon
    batch_size = args.batch_size
    method = args.method
    ila_layer = args.ila_layer
    linbp_layer = args.linbp_layer
    save_dir = args.save_dir
    niters = args.niters
    ila_niters = args.ila_niters
    target_attack = args.target_attack

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    model = models.__dict__['vgg19_bn'](num_classes=10)
    model.features = nn.DataParallel(model.features)
    model.load_state_dict(torch.load('attack/cifar10/models/ckpt/vgg19_bn/model_best.pth.tar', map_location=(device))['state_dict'])

    model = nn.Sequential(
        Normalize(),
        model
    )
    model.to(device)
    model.eval()

    cifar10 = SelectedCifar10('data/cifar10/cifar-10-batches-py',
                              'data/cifar10/selected_cifar10.csv',
                              transform=T.ToTensor())
    ori_loader = torch.utils.data.DataLoader(cifar10, batch_size=batch_size, shuffle=False, num_workers = 8)

    if target_attack:
        label_switch = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).long()
    label_ls = []
    for ind, (ori_img, label)in enumerate(ori_loader):
        label_ls.append(label)
        if target_attack:
            label = label_switch[label]
        ori_img = ori_img.to(device)
        img = ori_img.clone()
        m = 0
        for i in range(niters):
            # In our implementation of PGD, we incorporate randomness at each iteration to further enhance the transferability
            elif 'pgd' in method:
                img_x = img + img.new(img.size()).uniform_(-epsilon, epsilon)
            else:
                img_x = img
            img_x.requires_grad_(True)
            if 'linbp' in method:
                output = vgg19_forw(model, input_diversity(img_x) if method == 'mdi2fgsm' or method == 'linbp_mdi2fgsm' else img_x, True, linbp_layer)
            else:
                output = vgg19_forw(model, input_diversity(img_x) if method == 'mdi2fgsm' or method == 'linbp_mdi2fgsm' else img_x, False, None)
            loss = nn.CrossEntropyLoss()(output, label.to(device))
            model.zero_grad()
            loss.backward()
            if 'mdi2fgsm' in method or 'mifgsm' in method:
                g = img_x.grad.data
                input_grad = 1 * m + g / torch.norm(g, dim=(1, 2, 3), p=1, keepdim=True)
                m = input_grad
            else:
                input_grad = img_x.grad.data
            if target_attack:
                input_grad = -input_grad
            if method == 'fgsm' or '_fgsm' in method:
                img = img.data + 2 * epsilon * torch.sign(input_grad)
            else:
                img = img.data + 1./255 * torch.sign(input_grad)
            img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
            img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
            img = torch.clamp(img, min=0, max=1)

        if 'ila' in method:
            m = 0
            attack_img = img.clone()
            img = ori_img.clone().to(device)
            with torch.no_grad():
                mid_output = vgg19_ila_forw(model, ori_img, ila_layer)
                mid_original = torch.zeros(mid_output.size()).to(device)
                mid_original.copy_(mid_output)

                mid_output = vgg19_ila_forw(model, attack_img, ila_layer)
                mid_attack_original = torch.zeros(mid_output.size()).to(device)
                mid_attack_original.copy_(mid_output)

            for _ in range(ila_niters):
                img.requires_grad_(True)
                mid_output = vgg19_ila_forw(model, img, ila_layer)

                loss = ILAProjLoss()(
                    mid_attack_original.detach(), mid_output, mid_original.detach(), 1.0
                )

                model.zero_grad()
                loss.backward()
                input_grad = img.grad.data
                if method == 'ila_fgsm':
                    img = img.data + 2 * epsilon * torch.sign(input_grad)
                else:
                    img = img.data + 1./255 * torch.sign(input_grad)
                img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
                img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
                img = torch.clamp(img, min=0, max=1)
        np.save(save_dir + '/batch_{}.npy'.format(ind), torch.round(img.data*255).cpu().numpy())
        print('batch_{}.npy saved'.format(ind))
    label_ls = torch.cat(label_ls)
    np.save(save_dir + '/labels.npy', label_ls.numpy())
    print('all batches saved')
