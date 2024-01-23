import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
# from archs.cifar_resnet import resnet as resnet_cifar
# from src.img2img.img2img_data import get_normalize_layer
from torch.nn.functional import interpolate

import math


import torch.nn as nn
import torch.nn.functional as F
# from src.img2img.edsr import EDSR, SEDSR
# from src.img2img.dnet import DResNet
from src.img2img.unet import UNet, MUNet
from src.img2img.resunet import ResUnet

ARCHITECTURES = ["edsr","unet"]



_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_MINIST_MEAN = [0.1307]
_MINIST_STDDEV = [0.3081]

def img2img_get_model(args, img_size, param_size, parallel=False) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if args.arch == 'edsr':
        model = EDSR(args,img_size,param_size)

    elif args.arch == 'unet':
        if args.dataset == "mnist":
            model = MUNet(args, img_size, param_size)
        else:
            model = UNet(args,img_size,param_size)

    elif args.arch == 'runet':
        model = ResUnet(args,img_size,param_size)

    elif args.arch == 'sedsr':
        model = SEDSR(args,img_size,param_size)

    elif args.arch == 'cifar_resnet20':
        model = DResNet(depth=20,num_classes=args.n_bins)
    else:
        raise NotImplementedError

    if parallel == True:
        model = torch.nn.DataParallel(model)
        print('Using parallel training')
#     model = model.cuda()
    # normalize_layer = get_normalize_layer(dataset)
    return model




def get_size(args):

    if args.dataset=='cifar10':
        img_size = 32
    elif args.dataset == "cifar100":
        img_size = 32
    elif args.dataset == "tiny_imagenet":
        img_size = 48
    elif args.dataset == "mnist":
        img_size = 28
    else:
        raise NotImplementedError

    if args.corrupt == 'gaussian_blur' or args.corrupt=='zoom_blur' or args.corrupt=='rotate' or args.corrupt=='none' or args.corrupt=='contrast' or args.corrupt=='pixelate' or args.corrupt=='jpeg'\
            or args.corrupt == "rotational_blur" or args.corrupt == "zoom_blur" or args.corrupt == "defocus_blur" or args.corrupt == "scaling":
        param_size = 1
    elif args.corrupt == 'motion_blur' or args.corrupt=='translate':
        param_size = 2

    elif args.corrupt == 'gaussian_noise':
        param_size = args.n_bins


    else:
        raise NotImplementedError

    return img_size, param_size

