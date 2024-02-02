
import sys
sys.path.append('./..')
sys.path.append('./img2img')

import os

import argparse
import datetime

from time import time
from collections import OrderedDict
import yaml

import click
import numpy as np
import torch
from torchvision.models.resnet import resnet50
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.datasets import get_dataset, DATASETS, get_num_classes, get_corrupted_dataset, load_parallel_model
from src.img2img.img2img_models import img2img_get_model, get_size
from src.core import Smooth, TSmooth
from src.img2img.corruption import Corruption
from src.architectures import get_architecture
from some_our_code.utils import make_our_dataset_v2, CustomAudioDataset
from archs.cifar_resnet import resnet as resnet_cifar

sys.path.append('./../../')
# from architectures import get_architecture
from datasets_utils import get_normalize_layer


NUM_IMAGES_FOR_TEST = 500


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset",
                    default='cifar10',choices=DATASETS, help="which dataset")
parser.add_argument("--base_classifier", type=str,
                    default=None,
                    help="path to saved pytorch model of base classifier")
# parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str,
                    default='./logs/certify_results',
                    help="output file")
parser.add_argument("--batch", type=int, default=512, help="batch size")
parser.add_argument("--skip", type=int, default=30
                    , help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=200)
parser.add_argument("--N", type=int, default=1000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=1e-3, help="failure probability")
parser.add_argument('--corrupt',type=str,default=['none','gaussian_blur','motion_blur','zoom_blur','rotate','translate','contrast','pixelate','jpeg',][1],
                    help=' The corruption type for training')
parser.add_argument('--add_noise',type=float, default=0.0)
parser.add_argument('--noise_dst',default=["none","gaussian","exp","uniform","folded_gaussian"][2],type=str)
parser.add_argument('--noise_sd', default=0.8, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--partial_min',default=0.0, type=float,
                    help = "Minimal of certify range")
parser.add_argument('--partial_max',default=1.0, type=float,
                    help = "Maximal of certify range")
parser.add_argument('--arch', type=str, default=["edsr","unet","runet"][2])
args = parser.parse_args([])


def make_test_dataset(args):
    test_dataset = get_dataset(args.dataset, "test")
    pin_memory = (args.dataset == "imagenet")
    np.random.seed(42)
    idxes = np.random.choice(len(test_dataset), NUM_IMAGES_FOR_TEST, replace=False)
    
    ourdataset = make_our_dataset_v2(test_dataset, idxes)
    ourdataloader = DataLoader(ourdataset, shuffle=False, batch_size=1,
                         num_workers=6, pin_memory=False)
    return ourdataset, ourdataloader



def calculate_gsmooth(args):
    
    device = torch.device(args.device)
    model = get_architecture(arch=args.arch, dataset=args.dataset, device=device)
    checkpoint = torch.load(args.base_classifier, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)

    base_classifier = model
    
    dataset, _ = make_test_dataset(args)
    
    base_classifier.to(device)
    base_classifier.eval()

    corruptor = Corruption(args, co_type=args.corrupt,add_noise=args.add_noise,noise_sd=args.noise_sd,distribution=args.noise_dst)

    # create the smooothed classifier g
    smoothed_classifier = TSmooth(base_classifier, None, corruptor, get_num_classes(args.dataset),args.noise_dst,args.noise_sd, args.add_noise)

    # prepare output file
    filename = args.outfile+'_'+args.dataset+'_'+args.corrupt+'_'+str(args.noise_sd) +"_" +str(args.partial_max)
    f = open(filename+'_running', 'w')
    print("idx\tlabel\tpredict\tradius\tgood\tcorrect\ttime", file=f, flush=True)
    print("idx\tlabel\tpredict\tradius\tgood\tcorrect\ttime")

    tot, tot_good, tot_correct = 0, 0, 0

    # for gaussian smooth
    attack_radius = args.partial_max
    for i in tqdm(range(len(dataset))):
        (x, label) = dataset[i]

        before_time = time()
        x = x.to(device)
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)


        correct = (prediction == label).item()
        cond1 = radius * args.noise_sd > args.partial_max
        good = (radius * args.noise_sd > args.partial_max)&correct

        tot, tot_good, tot_correct = tot+1, tot_good+good, tot_correct+correct
        after_time = time()
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.5f}\t{}\t{}\t{}".format(i, label, prediction, radius, good, correct, time_elapsed), file=f, flush=True)
        print("{}\t{}\t{}\t{:.5f}\t{}\t{}\t{}".format(i, label, prediction, radius, good, correct, time_elapsed))

    f.close()

    print("Total {} Certified {} Certified Acc {} Test Acc {}".format(tot, tot_good, tot_good/tot, tot_correct/tot))

    f = open(filename+'_total_result', 'w')
    print("Total {} Certified {} Certified Acc {} Test Acc {}".format(tot, tot_good, tot_good/tot, tot_correct/tot), file=f, flush=True)
    f.close()

    
@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    args.corrupt = config["corrupt"]
    args.noise_sd = config["noise_sd"]
    args.noise_dst = config["noise_dst"]
    args.partial_max = config["partial_max"]
    args.dataset = config["dataset"]
    args.base_classifier = config["base_classifier"]
    args.device = config["device"]
    args.arch = config["arch"]

    calculate_gsmooth(args)

if __name__ == "__main__":
    main()