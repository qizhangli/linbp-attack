# LinBP-Attack
Code for our NeurIPS 2020 paper [Backpropagating Linearly Improves Transferability of Adversarial Examples](https://arxiv.org/abs/2012.03528).

## Environments
* Python 3.7.4
* PyTorch 1.4.0
* Torchvision 0.5.0
* Pillow 7.0.0
* Numpy 1.18.1

## Datasets
CIFAR-10 and ImageNet should be prepared into the following structure:
```
linbp-attack
└───data
    ├── cifar10
    │   ├── selected_cifar10.csv
    │   └── cifar-10-batches-py
    └── imagenet
        ├── selected_imagenet.csv
        └── ILSVRC2012_img_val
```
For CIFAR-10, select images from validation set, and write ```selected_cifar10.csv``` as following:
```
class_index,data_index
3,0
8,1
8,2
...
```
The ```data_index``` is the index of image in ```data/cifar10/cifar-10-batches-py/test_batch```

For ImageNet, select images from validation set, and write ```selected_imagenet.csv``` as following:
```
class_index, class, image_name
0,n01440764,ILSVRC2012_val_00002138.JPEG
2,n01484850,ILSVRC2012_val_00004329.JPEG
...
```

## Models
Please download pretrained models at [here](https://drive.google.com/drive/folders/1WqIxgfD7V4A9pbJNK2E2FJA5rSZqbXcP?usp=sharing), 
then extract them to ```attack/cifar10/models``` and ```attack/imagenet/models```.

## Attack
Untarget attack on ***CIFAR-10*** using PGD+LinBP+ILA, under epsilon = 0.03 :
```
python3 attack/cifar10/attack_vgg19.py --epsilon 0.03 --niters 100 --ila_niters 100 --method linbp_ila_pgd \
--save_dir data/cifar10/linbp_ila_pgd --batch_size 500
```
Untarget attack on ***ImageNet*** using PGD+LinBP+ILA, under epsilon = 0.03 :
```
python3 attack/imagenet/attack_resnet50.py --epsilon 0.03 --niters 300 --ila_niters 100 --method linbp_ila_pgd \
--save_dir data/imagenet/linbp_ila_pgd --batch_size 50 --sgm_lambda 1.0
```
```--sgm_lambda``` is the scaling factor in the residual units of SGM method.\
For both CIFAR-10 and ImageNet, add ```--target_attack``` to mount a target attack.
The ```--method``` can be set as ```fgsm/ifgsm/pgd/mifgsm/mdi2fgsm```. Either ILA, LinBP or both are supported by adding ```ila/linbp/linbp_ila``` like ```linbp_pgd```.
## Test
***CIFAR-10***:
```
python3 attack/cifar10/test.py --dir data/cifar10/linbp_ila_pgd
```
***ImageNet***:
```
python3 attack/imagenet/test.py --dir data/imagenet/linbp_ila_pgd
```
## Acknowledgements
The following resources are very helpful for our work:\
Pretrained models for CIFAR-10: [bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification)\
Pretrained models for ImageNet: [Cadene/pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) and [torchvision.models](https://pytorch.org/docs/stable/torchvision/models)\
GDAS: [D-X-Y/AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects)\
Pretrained GDAS: [ZiangYan/subspace-attack.pytorch](https://github.com/ZiangYan/subspace-attack.pytorch)

## Citation
```
@inproceedings{guo2020backpropagating,
    title={Backpropagating Linearly Improves Transferability of Adversarial Examples.},
    author={Guo, Yiwen and Li, Qizhang and Chen, Hao},
    booktitle={NeurIPS},
    year={2020}
}
```
