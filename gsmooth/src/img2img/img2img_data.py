from torchvision import transforms, datasets
from typing import *
from PIL import Image
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from src.img2img.corruption import Corruption

IMAGENET_LOC_ENV = "IMAGENET_DIR"
# list of all datasets
DATASETS = ["imagenet", "cifar10", "mnist"]


def img2img_dataset(args, dataset: str, split: str, corrupt: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split, corrupt)
    elif dataset == "cifar10":
        return _cifar10(args,split, corrupt)
    elif dataset == "cifar100":
        return _cifar100(args, split, corrupt)
    elif dataset == "tiny_imagenet":
        return  _tiny_imagenet(args, split, corrupt)
    elif dataset == "mnist":
        return _mnist(args, split, corrupt)



def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "cifar100":
        return 100
    elif dataset == "mnist":
        return 10


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "cifar100":
        return NormalizeLayer(_CIFAR100_MEAN, _CIFAR100_STDDEV)
    elif dataset == "mnist":
        return NormalizeLayer(_MINIST_MEAN, _MINIST_STDDEV)
    elif dataset == "tiny_imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


_CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
_CIFAR100_STDDEV = [0.2675, 0.2565, 0.2761]


_MINIST_MEAN = [0.1307]
_MINIST_STDDEV = [0.3081]

def _mnist(args, split:str, corrupt: str)->Dataset:
    if split == "train":
        return Img2img_MNIST(args, "./../dataset_cache", corruption_type=corrupt, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))
    elif split == "test":
        return Img2img_MNIST(args,"./../dataset_cache", corruption_type=corrupt,train=False, download=True, transform=transforms.ToTensor())




def _cifar10(args, split: str, corrupt: str) -> Dataset:
    if split == "train":
        return Img2img_CIFAR10(args, "./../dataset_cache", corruption_type=corrupt,train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    elif split == "test":
        return Img2img_CIFAR10(args,"./../dataset_cache", corruption_type=corrupt,train=False, download=True, transform=transforms.ToTensor())


def _cifar100(args, split: str, corrupt: str)-> Dataset:
    if split == "train":
        return Img2img_CIFAR100(args, "./../dataset_cache", corruption_type=corrupt,train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    elif split == "test":
        return Img2img_CIFAR100(args,"./../dataset_cache", corruption_type=corrupt,train=False, download=True, transform=transforms.ToTensor())



def _imagenet(split: str, corrupt: str) -> Dataset:
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

def create_val_img_folder():
    """
    This method is responsible for separating validation images into separate sub folders.
    """
    val_dir = './dataset_cache/tiny-imagenet-200/val'
    img_dir = os.path.join(val_dir, 'images')

    file = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = file.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    file.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

def _tiny_imagenet(args, split: str, corrupt)->Dataset:
    tiny_imagenet_dir = "./dataset_cache/tiny-imagenet-200"
    create_val_img_folder()
    if split == "train":

        data = Img2img_TinyImageNet(args, root= tiny_imagenet_dir+ '/train', transform=transforms.Compose([
        transforms.Resize((32,32)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]), corruption_type=corrupt)
    elif split == "test":
        data = Img2img_TinyImageNet(args,root=tiny_imagenet_dir + '/val/images', transform=transforms.Compose([
        transforms.Resize((32,32)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]), corruption_type = corrupt)
    else:
        raise ValueError

    return data


class NormalizeLayer(torch.nn.Module):


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



class Img2img_MNIST(datasets.MNIST):
    def __init__(
            self,
            args,
            root: str,
            corruption_type: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,

    ):
        super(Img2img_MNIST, self).__init__(root, train, transform, target_transform, download)

        self.corruptor = Corruption(args, co_type=corruption_type, )

    # input [0,255], output[0,1], shape b*h*w*c
    def __getitem__(self, index):
        img = self.data[index]

        # img = Image.fromarray(img)
        img = img.unsqueeze(-1).numpy()
        target, params = self.corruptor.apply(img)

        # to tensor will scale [0,255] to [0,1] and keeps [0,1] unchanged
        img, target, params = self.transform(img).float(), self.transform(target).float(), torch.from_numpy(
            params).float()

        return (img, params), target


class Img2img_CIFAR10(datasets.CIFAR10):
    def __init__(
            self,
            args,
            root: str,
            corruption_type: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,

    ):
        super(Img2img_CIFAR10, self).__init__(root, train, transform, target_transform, download)

        self.corruptor = Corruption(args,co_type=corruption_type,)

        # self.transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor()
        # ])


    #input [0,255], output[0,1], shape b*h*w*c
    def __getitem__(self,index):
        img = self.data[index]

        # img = Image.fromarray(img)
        target, params = self.corruptor.apply(img)

        # to tensor will scale [0,255] to [0,1] and keeps [0,1] unchanged
        img, target, params = self.transform(img).float(), self.transform(target).float(), torch.from_numpy(params).float()

        return (img, params) , target


class Img2img_CIFAR100(datasets.CIFAR100):
    def __init__(
            self,
            args,
            root: str,
            corruption_type: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,

    ):
        super(Img2img_CIFAR100, self).__init__(root, train, transform, target_transform, download)

        self.corruptor = Corruption(args,co_type=corruption_type,)

        # self.transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor()
        # ])


    #input [0,255], output[0,1], shape b*h*w*c
    def __getitem__(self,index):
        img = self.data[index]

        # img = Image.fromarray(img)
        target, params = self.corruptor.apply(img)

        # to tensor will scale [0,255] to [0,1] and keeps [0,1] unchanged
        img, target, params = self.transform(img).float(), self.transform(target).float(), torch.from_numpy(params).float()

        return (img, params) , target




class Img2img_TinyImageNet(datasets.ImageFolder):
    def __init__(
            self,
            args,
            root,
            corruption_type: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,

    ):
        super(Img2img_TinyImageNet, self).__init__(root,transform=transform)

        # if train:
        #     self.dataset = _tiny_imagenet(split='train')
        # else:
        #     self.dataset = _tiny_imagenet(split="test")
        self.corruptor = Corruption(args,co_type=corruption_type,)

        # self.transform = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor()
        # ])
        self.to_tensor = transforms.ToTensor()


    #input [0,255], output[0,1], shape b*h*w*c
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        img = self.loader(path) # a PIL object

        img = self.transform(img)   # a PIL object, resized image
        # img = Image.fromarray(img)
        target, params = self.corruptor.apply(np.array(img))    #PIL->array [0,255] RGB h*w*c

        # to tensor will scale [0,255] to [0,1] and keeps [0,1] unchanged
        img, target, params = self.to_tensor(img).float(), self.to_tensor(target).float(), torch.from_numpy(
            params).float()


        return (img, params), target
