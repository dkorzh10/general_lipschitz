import os
import sys

os.environ['WANDB_API_KEY'] = 'de1ceccfa66f1ba2bad667f4a2fd7666be9813fb'
os.environ['WANDB_PROJECT'] = 'lipschitz'
sys.path.append('./..')

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import scipy
import itertools
import equinox
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import kornia
import math
import imgaug
import argparse
import torch.nn.functional as F
import wandb
import datetime

from tqdm import tqdm
from time import time
from csaps import csaps
from scipy.stats import norm, binom_test
from math import ceil, sqrt
from statsmodels.stats.proportion import proportion_confint
from scipy import stats
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

from train_utils import AverageMeter, accuracy, init_logfile, log
from src.smoothing_and_attacks import construct_phi
from datasets_utils import get_dataset, DATASETS, get_normalize_layer
from architectures import ARCHITECTURES, get_architecture
from torchvision.models import resnet50, resnet101

models = {
        'resnet50':resnet50, 
          'resnet101':resnet101
          }




def kl_div(input, targets, reduction='batchmean'):
    return F.kl_div(F.log_softmax(input, dim=1), targets, reduction=reduction)


def _cross_entropy(input, targets, reduction='mean'):
    targets_prob = F.softmax(targets, dim=1)
    xent = (-targets_prob * F.log_softmax(input, dim=1)).sum(1)
    if reduction == 'sum':
        return xent.sum()
    elif reduction == 'mean':
        return xent.mean()
    elif reduction == 'none':
        return xent
    else:
        raise NotImplementedError()


def _entropy(input, reduction='mean'):
    return _cross_entropy(input, input, reduction)




def _chunk_minibatch(batch, num_batches):
    X, y = batch
    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]


#--------------------------------------------------------------------------------------------------------------------------


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer, epoch: int,
          transformer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_reg = AverageMeter()
    confidence = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time()

    # switch to train mode
    model.train()

    for i, batch in enumerate(tqdm(loader)):
        # measure data loading time
        data_time.update(time() - end)

        mini_batches = _chunk_minibatch(batch, args.num_noise_vec)
        for inputs, targets in mini_batches:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            batch_size = inputs.size(0)

            noised_inputs = [transformer(inputs.to(args.device)) for _ in range(args.num_noise_vec)]

            # augment inputs with noise
            inputs_c = torch.cat(noised_inputs, dim=0)
            targets_c = targets.repeat(args.num_noise_vec)

            logits = model(inputs_c)

            loss_xent = criterion(logits, targets_c)

            logits_chunk = torch.chunk(logits, args.num_noise_vec, dim=0)
            softmax = [F.softmax(logit, dim=1) for logit in logits_chunk]
            avg_softmax = sum(softmax) / args.num_noise_vec

            consistency = [kl_div(logit, avg_softmax, reduction='none').sum(1)
                           + _entropy(avg_softmax, reduction='none')
                           for logit in logits_chunk]
            consistency = sum(consistency) / args.num_noise_vec
            consistency = consistency.mean()

            loss = loss_xent + args.lbd * consistency

            avg_confidence = -F.nll_loss(avg_softmax, targets)

            acc1, acc5 = accuracy(logits, targets_c, topk=(1, 5))
            losses.update(loss_xent.item(), batch_size)
            losses_reg.update(consistency.item(), batch_size)
            confidence.update(avg_confidence.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time() - end)
        end = time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            
            torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, args.type + args.expname+'_checkpoint.pth.tar'))
            
            
            wandb.log({
                f'epoch/{epoch}/loss/train': losses.avg,
                f'epoch/{epoch}/loss/consistency': losses_reg.avg,
                f'epoch/{epoch}/loss/avg_confidence': confidence.avg,
                f'epoch/{epoch}/batch_time': batch_time.avg,
                f'epoch/{epoch}/accuracy/train@1': top1.avg,
                f'epoch/{epoch}/accuracy/train@5': top5.avg
                
            })

    
    wandb.log({
        "epoch": epoch,
        'loss/train': losses.avg,
        'loss/consistency': losses_reg.avg,
        'loss/avg_confidence': confidence.avg,
        'batch_time': batch_time.avg,
        'accuracy/train@1': top1.avg,
        'accuracy/train@5': top5.avg
        
    })

    return (losses.avg, top1.avg)


def test(loader, model, criterion, transformer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(loader)):
            # measure data loading time
            data_time.update(time() - end)

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            # augment inputs with noise
            inputs = transformer(inputs.to(args.device))

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    i, len(loader), batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1, top5=top5))

        wandb.log({
            'loss/test': losses.avg,
            'accuracy/test@1': top1.avg,
            'accuracy/test@5': top5.avg
        
        })


        return (losses.avg, top1.avg)



#---------


    
#-----------



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help="type of transform", type=str, default='cb')
    
    
#     python train.py --run_name cifar100_trans  --dataset cifar100 --arch cifar100_resnet110 --type tr --batch 256 --device cuda:0 --lr 0.001 --tr 20.0 
    
    
    
    
    parser.add_argument('--dataset', help="dataset name", type=str, choices=['imagenet', 'cifar10', 'mnist', 'cifar100'], default='cifar100')
    parser.add_argument('--epochs', help="n epochs", type=int, default=90)
    parser.add_argument('--batch', help="batch_size", type=int, default=512)
    parser.add_argument('--device', help="device", type=str, default='cuda:0')
    parser.add_argument('--seed', help="seed", type=int, default=0)
#     parser.add_argument('--results_dir', help="dir for results", type=str, default='./results')
    parser.add_argument('--lr', help="learning rate", type=float, default=1e-2)
    parser.add_argument('--momentum', help="momentum factor", type=float, default=.95)
    parser.add_argument('--wd', help="weight decay", type=float, default=1e-4)
    parser.add_argument('--lr_step_size', help="learning rate step size", type=float, default=20)
    parser.add_argument('--gamma', help="gamma????", type=float, default=.1)
    parser.add_argument('--arch', help="architecture of nn to train", type=str, choices=['resnet50', 'cifar_resnet110', 'cifar100_resnet110'], default='cifar100_resnet110')
    parser.add_argument('--workers', help="n workers", type=int, default=8)
    parser.add_argument('--pin_memory', help="pin mem?", type=bool, default=False)
    parser.add_argument('--outdir', help="direct where to save sht", type=str, default='./new_results/')
    parser.add_argument('--num_noise_vec', help='i dont know wtf is this', type=str, default=1)
    parser.add_argument('--lbd', help="LGBT+++-*", type=int, default=10)
    parser.add_argument('--print_freq', help="how hard it is 4 me to shake the disease", type=int, default=10)
    parser.add_argument('--print_step', type=int, default=10)
    
    
    ## ----- ----- params 4 transformer
    parser.add_argument('--sigma', help="variance of additive noise", type=float, default=.8)
    parser.add_argument('--tau', help="TAU", type = int, default=50)
    parser.add_argument('--tr', help="sigma of translation parameter", type=float, default=15.)
    parser.add_argument('--sigma_b', help="sigma of brightnesf  parameter", type=float, default=.4)
    parser.add_argument('--sigma_bl', help="sigma of blur gaussian", type=float, default=10.)
    parser.add_argument('--sigma_c', help="-==-== of contrast", type=float, default=.4)
    
    parser.add_argument('--run_name', required=True, help="run name", type=str)

    args = parser.parse_args()
    
    expname = args.arch + '_' + args.dataset + '_' + args.type + '_' + args.run_name
    args.expname = expname
    

    
    
    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                            num_workers = args.workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                            num_workers = args.workers, pin_memory=args.pin_memory)
    
    
    
    ## model you know
    
    model = get_architecture(arch=args.arch, dataset=args.dataset, device=args.device)
    model = model.to(args.device)
    
    #guys needed for tarining
    criterion = CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)





#     transformer = phi_tbbc_torch_batch_and_noise
    transformer = construct_phi(tr_type = args.type, 
                            device=args.device,
                            sigma_b=args.sigma_b, 
                            sigma_c=args.sigma_c, 
                            sigma_tr=args.tr, sigma_blur=args.sigma_bl)
    
    
    wandb.init(project="lipschitz")
    
    #spray and pray:
    
    for epoch in range(args.epochs):
        before = time()
        train_loss, train_acc = train(loader=train_loader, 
                                      model=model, 
                                      criterion=criterion, 
                                      optimizer=optimizer, 
                                      transformer=transformer, 
                                      epoch=epoch,
                                      args=args)
        
        test_loss, test_acc = test(loader=test_loader, 
                                   model=model, 
                                   criterion=criterion, 
                                   transformer=transformer, 
                                   args=args)
        after = time()

        scheduler.step(epoch)

    
        print("{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, expname+'_checkpoint.pth.tar'))

