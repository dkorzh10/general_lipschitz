from torchvision import transforms, datasets
from typing import *
from src.img2img.corruption import Corruption
import torch
import os
from torch.utils.data import Dataset
from collections import OrderedDict
import numpy as np

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"
os.environ[IMAGENET_LOC_ENV] = "/workspace/mnt/local/dataset/by-domain/cv/imagenet"

# list of all datasets
DATASETS = ["imagenet", "cifar10", "mnist"]






def get_dataset(dataset: str, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    elif dataset == "mnist":
        return _mnist10(split)
    elif dataset == "cifar10":
        return _cifar10(split)
    elif dataset == "cifar100":
        return _cifar100(split)
    elif dataset == "tiny_imagenet":
        return _tiny_imagenet(split)



def get_corrupted_dataset(dataset: str, split: str, corrupt_param=None) -> Dataset:


    if corrupt_param == None:
        corrupt_param = {'type':'none','add_noise':0,'sd':1,'dst':'none'}
    corruptor = Corruption(None,co_type=corrupt_param['type'],add_noise=corrupt_param['add_noise'],noise_sd=corrupt_param['sd'],distribution=corrupt_param['dst'])

    if split == "train":
        is_train = True
    else:
        is_train = False


    if dataset == "imagenet":
        raise NotImplementedError

    elif dataset == "mnist":
        return CorruptMNIST("./dataset_cache", corruptor=corruptor, train=is_train, download=True,
                              transform=transforms.Compose([
                                  # transforms.RandomCrop(32, padding=4),
                                  # transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor()]))

    elif dataset == "cifar10":

        return CorruptCifar10("./dataset_cache",corruptor=corruptor, train=is_train, download=True, transform=transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))
    elif dataset == "cifar100":
        return CorruptCifar100("./dataset_cache", corruptor=corruptor, train=is_train, download=True,
                              transform=transforms.Compose([
                                  # transforms.RandomCrop(32, padding=4),
                                  # transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor()]))

    elif dataset == "tiny_imagenet":
        return CorruptTinyImageNet("/img2img/dataset_cache/tiny-imagenet-200",corruptor=corruptor, train=is_train,
                                   transform=transforms.Compose([transforms.ToTensor()]))


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "mnist":
        return 10
    elif dataset == "cifar10":
        return 10
    elif dataset == "cifar100":
        return 100
    elif dataset == "tiny_imagenet":
        return 200
    else:
        raise NotImplementedError


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "mnist":
        return NormalizeLayer(_MINIST_MEAN, _MINIST_STDDEV)
    elif dataset == "cifar100":
        return NormalizeLayer(_CIFAR100_MEAN, _CIFAR100_STDDEV)

    elif dataset == "tiny_imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    else:
        raise NotImplementedError


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
_CIFAR100_STDDEV = [0.2675, 0.2565, 0.2761]

_MINIST_MEAN = [0.1307]
_MINIST_STDDEV = [0.3081]

def _mnist10(split:str)->Dataset:
    if split == "train":
        return datasets.MNIST("./dataset_cache", train=True, download=True, transform=transforms.ToTensor())
    elif split == "test":
        return datasets.MNIST("./dataset_cache",train=False,download=True, transform=transforms.ToTensor())



def _cifar10(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


def _cifar100(split: str) -> Dataset:
    if split == "train":
        return datasets.CIFAR100("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR100("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())

def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


def _tiny_imagenet(split: str)->Dataset:
    tiny_imagenet_dir = "./img2img/dataset_cache/tiny-imagenet-200"
    if split == "train":

        data = datasets.ImageFolder(root= tiny_imagenet_dir+ '/train', transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]))
    elif split == "test":
        data = datasets.ImageFolder(root=tiny_imagenet_dir + '/val/images', transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]))
    else:
        raise ValueError

    return data

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds


class CorruptMNIST(datasets.MNIST):
    def __init__(self,root, corruptor, train, transform, target_transform=None, download=True):
        super(CorruptMNIST, self).__init__(root, train, transform, target_transform, download)

        self.corruptor = corruptor


    #input [0,255], output[0,1], shape b*h*w*c
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = img.unsqueeze(-1).numpy()
        img_corrupt, params = self.corruptor.apply(img)

        img_corrupt, target, params = self.transform(img_corrupt).float(), torch.LongTensor([target]), torch.from_numpy(params).float()

        return (img_corrupt, params), target






class CorruptCifar10(datasets.CIFAR10):
    def __init__(self,root, corruptor, train, transform, target_transform=None, download=True):
        super(CorruptCifar10, self).__init__(root, train, transform, target_transform, download)

        self.corruptor = corruptor


    #input [0,255], output[0,1], shape b*h*w*c
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img_corrupt, params = self.corruptor.apply(img)

        img_corrupt, target, params = self.transform(img_corrupt).float(), torch.LongTensor([target]), torch.from_numpy(params).float()

        return (img_corrupt, params), target






class CorruptCifar100(datasets.CIFAR100):
    def __init__(self,root, corruptor, train, transform, target_transform=None, download=True):
        super(CorruptCifar100, self).__init__(root, train, transform, target_transform, download)

        self.corruptor = corruptor


    #input [0,255], output[0,1], shape b*h*w*c
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img_corrupt, params = self.corruptor.apply(img)

        img_corrupt, target, params = self.transform(img_corrupt).float(), torch.LongTensor([target]), torch.from_numpy(params).float()

        return (img_corrupt, params), target


class CorruptTinyImageNet(datasets.ImageFolder):
    def __init__(self,root, corruptor, train, transform, target_transform=None):
        super(CorruptTinyImageNet, self).__init__(root, train, transform, target_transform)
        self.corruptor = corruptor

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        img_corrupt, params = self.corruptor.apply(np.array(img))

        img_corrupt, target, params = self.transform(img_corrupt).float(),torch.LongTensor([target]), torch.from_numpy(params).float()

        return (img_corrupt, params), target


def load_parallel_model(model, state_dict, gpu=0):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[:2]+k[9:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model