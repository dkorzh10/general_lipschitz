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
import numpy as np
import datetime
from src.architectures import get_architecture

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset",
                    default='cifar100',choices=DATASETS, help="which dataset")
parser.add_argument("--base_classifier", type=str,
                          default=None,
                    help="path to saved pytorch model of base classifier")
# parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str,
                    default='./logs/certify_results',
                    help="output file")


parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.01, help="failure probability")




parser.add_argument('--corrupt',type=str,default=['none','gaussian_blur','motion_blur','rotational_blur','zoom_blur','defocus_blur','rotate','translate','contrast','pixelate','scaling'][0],
                    help=' The corruption type for training')

parser.add_argument('--add_noise',type=float,default=0.0,
                    help=' add an isotropic gaussian noise with sigma   ---using neural networks')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of for data transformation ---using neural networks")

parser.add_argument('--noise_tco',default=0.0,type=float,
                    help = "The standard deviation for data transformation-using corruptor")
parser.add_argument('--noise_aco',default=0.0,type=float,
                    help = "Add Gaussian noise using corruptor            -using corruptor")



parser.add_argument('--noise_dst',default=["none","gaussian","exp","uniform","folded_gaussian"][1],type=str)

parser.add_argument('--N_ps',default=10,type=int,
                    help = "Number of samples used for progressive sampling")
parser.add_argument('--partial_min',default=0, type=float,
                    help = "Minimal of certify range")
parser.add_argument('--partial_max',default=1, type=float,
                    help = "Maximal of certify range")

parser.add_argument('--arch', type=str,
                    default=["edsr","unet","runet"][1])
parser.add_argument('--gpu',type=int,
                    default=0)
args = parser.parse_args()



# print("setGPU: Setting GPU to: {}".format(5))
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
torch.cuda.set_device(args.gpu)
if __name__ == "__main__":
    # load the base classifier
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:'+str(args.gpu))


    checkpoint = torch.load(args.base_classifier, map_location={'cuda:0':'cuda:'+str(args.gpu)})
    # checkpoint = torch.load(args.base_classifier)


    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    if args.dataset == 'mnist':
        base_classifier.load_state_dict(checkpoint["state_dict"])
    else:
        base_classifier = load_parallel_model(base_classifier, checkpoint['state_dict'])

    img_size, param_size = get_size(args)
    margin = (args.partial_max - args.partial_min) / args.N_ps
    model_t = img2img_get_model(args, img_size, param_size,parallel=False)

    trans_path = None

    if trans_path is not None:
        state_dict = torch.load(trans_path, map_location={'cuda:0':'cuda:'+str(args.gpu)})['state_dict']


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
    if args.noise_sd<0.01 and args.add_noise<0.01:
        smoothed_classifier = TSmooth(base_classifier, None, corruptor, get_num_classes(args.dataset), args.noise_dst,args.noise_sd, args.add_noise)
    else:
        smoothed_classifier = TSmooth(base_classifier, model_t, None, get_num_classes(args.dataset),args.noise_dst,args.noise_sd, args.add_noise)

    # prepare output file
    f = open(args.outfile+'_'+args.corrupt+'_'+str(args.noise_sd), 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime")

    # iterate through the dataset
    # dataset =get_corrupted_dataset(args.dataset, args.split, corrupt_param={'type':args.corrupt,'dst':args.noise_dst,'sd':args.noise_sd})
    dataset = get_dataset(args.dataset, args.split)
    tot, tot_good, tot_correct = 0, 0, 0
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]


        before_time = time()
        # certify the prediction of g around x

        x_proc = []
        # prepare samples
        for j in range(args.N_ps):
            x_new,_ = corruptor.apply(x.cpu().numpy().transpose([1,2,0])*255, params=np.array([(args.partial_min+(args.partial_max - args.partial_min)*j/args.N_ps)]))
            x_proc.append(torch.from_numpy(x_new.transpose([2,0,1])))

        # calculate the error bound
        # prepare batched image and Parameters
        p_max = corruptor.param_max[args.corrupt]
        n_p = 100
        n_iter = 100
        x = x.cuda().unsqueeze(0).repeat([n_p, 1, 1, 1])
        p = torch.linspace(args.partial_min, args.partial_max, n_p).unsqueeze(1).cuda()
        reduced_grad_norm = gradnorm_batch_reduced(model_t, x, p, n_iter)



        gnorm = ((1 / args.noise_sd) ** 2 +   (reduced_grad_norm / args.add_noise) ** 2) ** 0.5
        # gnorm = (  (reduced_grad_norm / args.add_noise) ** 2) ** 0.5
        # gnorm = 1 # for resolvable

        print('reduced grad norm {:.5f} gnorm {}'.format(reduced_grad_norm, gnorm))
        # gnorm = 1


        # progressive sampling
        good = True
        correct = True
        for j in range(args.N_ps):
            x_new = x_proc[j].cuda()


            prediction, radius = smoothed_classifier.certify(x_new, args.N0, args.N, args.alpha, args.batch)
            print('radius @ {} : {}'.format(j,radius))
            if radius / gnorm < margin or prediction != label :
                good = False
                if prediction != label:
                    correct  = False
                break

        tot, tot_good, tot_correct = tot+1, tot_good+int(good), tot_correct+int(correct)



        after_time = time()

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}_certify\t{}\t{}".format(i, label, good, correct, time_elapsed))
        print("{}\t{}\t{}\t{}\t{}".format(i, label, good, correct, time_elapsed), file=f, flush=True)

    f.close()

    print('Total {} Certified {} Certified Acc {} Test Acc {}'.format(tot, tot_good, tot_good/ tot, tot_correct/tot))
