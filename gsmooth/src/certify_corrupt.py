# evaluate a smoothed classifier on a dataset
import sys
sys.path.append('./..')

import argparse
import os
# import setGPU
from src.datasets import get_dataset, DATASETS, get_num_classes, get_corrupted_dataset, load_parallel_model
from src.img2img.img2img_models import img2img_get_model, get_size
from src.core import Smooth, TSmooth
from time import time
from src.grad2n import gradnorm_batch, gradnorm_batch_reduced
from collections import OrderedDict
from src.img2img.corruption import Corruption
from src.analyze import plot_certified_accuracy, Line, ApproximateAccuracy
import torch
import datetime
from src.architectures import get_architecture


'''certify corruptions with good mathematical properties'''




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


parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=100, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")




parser.add_argument('--corrupt',type=str,default=['none','gaussian_blur','motion_blur','zoom_blur','rotate','translate','contrast','pixelate','jpeg',][1],
                    help=' The corruption type for training')
parser.add_argument('--add_noise',type=float, default=0.001)
parser.add_argument('--noise_dst',default=["none","gaussian","exp","uniform"][3],type=str)
parser.add_argument('--noise_sd', default=1.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")


parser.add_argument('--arch', type=str,
                    default=["edsr","unet","runet"][1])

args = parser.parse_args()



# print("setGPU: Setting GPU to: {}".format(5))
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == "__main__":
    # load the base classifier
    device = torch.device('cuda:0')


    checkpoint = torch.load(args.base_classifier)


    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier = load_parallel_model(base_classifier, checkpoint['state_dict'])


    img_size, param_size = get_size(args)

    model_t = img2img_get_model(args, img_size, param_size,parallel=False)


    trans_path = transform_model_path
    if trans_path is not None:
        state_dict = torch.load(trans_path)['state_dict']
        model_t.load_state_dict(state_dict)
    # model_t = load_parallel_model(model_t, state_dict)

    # to device
    base_classifier.to(device)
    base_classifier.eval()
    model_t.to(device)
    model_t.eval()


    #load dataset

    # corruptor
    corruptor = Corruption(args, co_type=args.corrupt,add_noise=args.add_noise,noise_sd=args.noise_sd,distribution=args.noise_dst)

    # create the smooothed classifier g
    smoothed_classifier = TSmooth(base_classifier, None, corruptor, get_num_classes(args.dataset),args.noise_dst,args.noise_sd, args.add_noise)

    # prepare output file
    f = open(args.outfile+'_'+args.corrupt+'_'+str(args.noise_sd), 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime")

    # iterate through the dataset
    # dataset =get_corrupted_dataset(args.dataset, args.split, corrupt_param={'type':args.corrupt,'dst':args.noise_dst,'sd':args.noise_sd})
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)



        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(i, label, prediction, radius, correct, time_elapsed))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
