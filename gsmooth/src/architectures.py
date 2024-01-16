import torch
from torchvision.models.resnet import resnet50
import torch.backends.cudnn as cudnn
from src.archs.cifar_resnet import resnet as resnet_cifar
from src.archs.cnn import CNN, Conv4FC3
from src.archs.preresnet import PreResNet
from src.datasets import get_normalize_layer
from torch.nn.functional import interpolate

# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["resnet50", "cifar_resnet20", "cifar_resnet110"]


def get_architecture(arch: str, dataset: str, parallel=False, num_classes=10) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if dataset == "mnist":
        # model = CNN()
        model = Conv4FC3()
    else:
        if arch == "resnet50" and dataset == "imagenet":
            model = resnet50(pretrained=False).cuda()
            cudnn.benchmark = True
        elif arch == "cifar_resnet20":
            model = resnet_cifar(depth=20, num_classes=10).cuda()
        elif arch == "cifar_resnet110":
            if dataset == "cifar10":
                model = resnet_cifar(depth=110, num_classes=10).cuda()
            elif dataset == "cifar100":
                model = PreResNet(110, num_classes=100).cuda()
            elif dataset == "tiny_imagenet":
                model = PreResNet(110, num_classes=200).cuda()
            else:
                raise NotImplementedError
        else:
            raise ValueError
    if parallel == True:
        model = torch.nn.DataParallel(model)
        print('Using parallel training')
    model = model.cuda()
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
