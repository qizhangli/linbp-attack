import os, sys
import torch
import torchvision.transforms as T
import torch.nn as nn
import argparse
import torch.nn.functional as F
import torchvision
import models as MODEL
from torch.backends import cudnn
import numpy as np
from utils import SelectedImagenet, Normalize, input_diversity, \
    linbp_forw_resnet50, linbp_backw_resnet50, ila_forw_resnet50, ILAProjLoss

parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default=0.03)
parser.add_argument('--sgm_lambda', type=float, default=1.0)
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--ila_niters', type=int, default=100)
parser.add_argument('--method', type=str, default = 'linbp_ila_pgd')
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--linbp_layer', type=str, default='3_1')
parser.add_argument('--ila_layer', type=str, default='2_3')
parser.add_argument('--save_dir', type=str, default = '')
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
    sgm_lambda = args.sgm_lambda


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    trans = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.ToTensor()
    ])
    dataset = SelectedImagenet(imagenet_val_dir='data/imagenet/ILSVRC2012_img_val',
                               selected_images_csv='data/imagenet/selected_imagenet.csv',
                               transform=trans
                               )
    ori_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = False)
    model = MODEL.resnet.resnet50(state_dict_dir ='attack/imagenet/models/ckpt/resnet50-19c8e357.pth')
    model.eval()
    model = nn.Sequential(
            Normalize(),
            model
        )
    model.to(device)

    if target_attack:
        label_switch = torch.tensor(list(range(500,1000))+list(range(0,500))).long()
    label_ls = []
    for ind, (ori_img, label)in enumerate(ori_loader):
        label_ls.append(label)
        if target_attack:
            label = label_switch[label]

        ori_img = ori_img.to(device)
        img = ori_img.clone()
        m = 0
        for i in range(niters):
            if method == 'mdi2fgsm' or method == 'linbp_mdi2fgsm':
                if i == 0:
                    img_x = img
                else:
                    img_x = input_diversity(img)
            # In our implementation of PGD, we incorporate randomness at each iteration to further enhance the transferability
            elif 'pgd' in method:
                img_x = img + img.new(img.size()).uniform_(-epsilon, epsilon)
            else:
                img_x = img
            img_x.requires_grad_(True)

            if 'linbp' in method:
                att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(model, img_x, True, linbp_layer)
                pred = torch.argmax(att_out, dim=1).view(-1)
                loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                model.zero_grad()
                input_grad = linbp_backw_resnet50(img_x, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp=sgm_lambda)
            else:
                att_out = model(img_x)
                pred = torch.argmax(att_out, dim=1).view(-1)
                loss = nn.CrossEntropyLoss()(att_out, label.to(device))
                model.zero_grad()
                loss.backward()
                input_grad = img_x.grad.data
            model.zero_grad()
            if 'mdi2fgsm' in method or 'mifgsm' in method:
                input_grad = 1 * m + input_grad / torch.norm(input_grad, dim=(1, 2, 3), p=1, keepdim=True)
                m = input_grad
            if target_attack:
                input_grad = - input_grad
            if method == 'fgsm' or '_fgsm' in method:
                img = img.data + 2 * epsilon * torch.sign(input_grad)
            else:
                img = img.data + 1./255 * torch.sign(input_grad)
            img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
            img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
            img = torch.clamp(img, min=0, max=1)
        if 'ila' in method:
            attack_img = img.clone()
            img = ori_img.clone().to(device)
            with torch.no_grad():
                mid_output = ila_forw_resnet50(model, ori_img, ila_layer)
                mid_original = torch.zeros(mid_output.size()).to(device)
                mid_original.copy_(mid_output)
                mid_output = ila_forw_resnet50(model, attack_img, ila_layer)
                mid_attack_original = torch.zeros(mid_output.size()).to(device)
                mid_attack_original.copy_(mid_output)
            for _ in range(ila_niters):
                img.requires_grad_(True)
                mid_output = ila_forw_resnet50(model, img, ila_layer)
                loss = ILAProjLoss()(
                    mid_attack_original.detach(), mid_output, mid_original.detach(), 1.0
                )
                model.zero_grad()
                loss.backward()
                input_grad = img.grad.data
                model.zero_grad()
                if method == 'ila_fgsm':
                    img = img.data + 2 * epsilon * torch.sign(input_grad)
                else:
                    img = img.data + 1./255 * torch.sign(input_grad)
                img = torch.where(img > ori_img + epsilon, ori_img + epsilon, img)
                img = torch.where(img < ori_img - epsilon, ori_img - epsilon, img)
                img = torch.clamp(img, min=0, max=1)
            del mid_output, mid_original, mid_attack_original
        np.save(save_dir + '/batch_{}.npy'.format(ind), torch.round(img.data*255).cpu().numpy().astype(np.uint8()))
        del img, ori_img, input_grad
        print('batch_{}.npy saved'.format(ind))
    label_ls = torch.cat(label_ls)
    np.save(save_dir + '/labels.npy', label_ls.numpy())
    print('images saved')