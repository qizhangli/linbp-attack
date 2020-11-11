import os
import os.path as osp
import torch

from models.gdas.lib.scheduler import load_config
from models.gdas.lib.scheduler import load_config
from models.gdas.lib.nas import model_types
from models.gdas.lib.nas import NetworkCIFAR as Network

__all__ = ['gdas']


def gdas(checkpoint_fname):
    checkpoint = torch.load(checkpoint_fname, map_location='cpu')
    xargs = checkpoint['args']
    config = load_config(os.path.join(osp.dirname(__file__), xargs.model_config))
    genotype = model_types[xargs.arch]
    class_num = 10

    model = Network(xargs.init_channels, class_num, xargs.layers, config.auxiliary, genotype)
    model.load_state_dict(checkpoint['state_dict'])
    return model
