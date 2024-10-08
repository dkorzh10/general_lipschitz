import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.fashionmnist_conv import Conv2FC2full, Conv2FC2simple
from archs.mnist_conv import Conv4FC3
from datasets_utils import get_normalize_layer
from torch.nn.functional import interpolate
import torch.nn as nn


# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110", 'mnist_43']

def get_architecture(arch: str, dataset: str, device) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
#         model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
#         cudnn.benchmark = True
        model = resnet50(pretrained=False).to(device)
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).to(device)
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).to(device)
    elif arch == "cifar100_resnet110":
        model = resnet_cifar(depth=110, num_classes=100).to(device)
    elif arch == "fashion_22full":
        model = Conv2FC2full()
        model = model.to(device)
    elif arch == "fashion_22simple":
        model = Conv2FC2simple().to(device)
    elif arch == "mnist_43":
        model = Conv4FC3().to(device)
    normalize_layer = get_normalize_layer(dataset, device=device)
    return torch.nn.Sequential(normalize_layer, model)
