# this file is based on src publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.
import sys
sys.path.append('./..')

import argparse
import os
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from src.datasets import get_dataset, DATASETS, get_corrupted_dataset
from src.img2img.img2img_models import img2img_get_model, get_size
from src.architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer, Adam
from torch.optim.lr_scheduler import StepLR
import time
import datetime
from src.train_utils import AverageMeter, sample_noise, accuracy, init_logfile, log
from torch.utils.tensorboard import SummaryWriter



parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str,
                    default='cifar100', choices=["imagenet", "cifar10", "cifar100", "mnist"])
parser.add_argument('--arch', type=str,
                    default="cifar_resnet110", choices=["resnet50", "cifar_resnet20", "cifar_resnet110"])
parser.add_argument('--outdir', type=str,
                    default='./logs/train_models',
                    help='folder to save model and training log')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=40,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.99, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')


parser.add_argument('--corrupt',type=str,default=['none','gaussian_blur','motion_blur','rotational_blur','zoom_blur','defocus_blur','rotate','translate','contrast','pixelate','jpeg','scaling'][0],
                    help=' The corruption type for training')



parser.add_argument('--add_noise',type=float,default=0.25,
                    help=' add an isotropic gaussian noise with sigma   ---using neural networks')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of for data transformation ---using neural networks")
parser.add_argument('--noise_dst',default=["none","gaussian","exp","uniform"][1],
                    help = "Noise distribution using neural networks    ---using neural networks")

parser.add_argument('--noise_tco',default=0.5,type=float,
                    help = "The standard deviation for data transformation-using corruptor")
parser.add_argument('--noise_aco',default=0.0,type=float,
                    help = "Add Gaussian noise using corruptor            -using corruptor")
parser.add_argument('--noise_dco',default=["none","gaussian","exp","uniform","folded_gaussian"][1],type=str,
                    help = " Noise distribution using corruptor           -using corruptor")

parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--gpu', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--parallel',default=True, type=bool,help='Use parallel training')
parser.add_argument('--use_tb', type=str,
                    default=True,
                    help='Use tensorboard')
args = parser.parse_args()
# print(args)
comment = '0.25_0.5_foldedgaussian_ab'
img_size, param_size = get_size(args)


def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # if corrupt by neural networks, use clean loaders, otherwise use corrupted loaders
    if args.noise_tco>0.01 or args.noise_aco>0.01:
        train_dataset = get_corrupted_dataset(args.dataset, 'train',corrupt_param = {'type':args.corrupt,'add_noise':args.noise_aco,'dst':args.noise_dco,'sd':args.noise_tco})
    else:
        train_dataset = get_corrupted_dataset(args.dataset, 'train',corrupt_param=None)

    test_dataset = get_corrupted_dataset(args.dataset, 'test',corrupt_param = {'type':args.corrupt,'add_noise':args.noise_aco,'dst':args.noise_dco,'sd':args.noise_tco})
    clean_testset = get_corrupted_dataset(args.dataset, 'test', corrupt_param=None)
    pin_memory = (args.dataset == "imagenet")

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch, num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=512, num_workers=args.workers, pin_memory=pin_memory)
    clean_testloader = DataLoader(clean_testset, shuffle=True, batch_size=512,num_workers=args.workers,pin_memory=pin_memory)

    cls_arch = args.arch
    model_t = None
    if args.add_noise>0.01:

        trans_path = None
        state_dict = torch.load(trans_path)['state_dict']

        args.arch = 'unet'
        model_t = img2img_get_model(args, img_size, param_size,parallel=args.parallel)
        model_t.module.load_state_dict(state_dict)


    model = get_architecture(cls_arch, args.dataset, args.parallel)

    logfilename = os.path.join(args.outdir, 'log') + time.strftime('_%m%d_%H_%M') + args.corrupt + '_' + comment +'.txt'

    # logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")
    args.outdir = args.outdir + time.strftime('/checkpoint_%m%d_%H_%M') + args.corrupt + '_' + comment +'.pth.tar'


    log(logfilename, str(args), prt=True)

    log(logfilename, prt=False)


    if args.use_tb:
        writer = SummaryWriter(log_dir='./logs/train'+time.strftime('/%m%d_%H_%M')+'_'+args.corrupt+comment,comment='trainning logs')
    else:
        writer = None

    criterion = CrossEntropyLoss().cuda()
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    for epoch in range(args.epochs):
        scheduler.step(epoch)
        before = time.time()
        train_loss, train_acc = train(train_loader, model, model_t, criterion, optimizer, epoch, writer)
        test_clean_loss, test_clean_acc = test(clean_testloader, model, criterion,epoch, writer, corrupt=False)
        test_corrupt_loss, test_corrupt_acc = test(clean_testloader, model, criterion,epoch , writer, corrupt=True)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_lr()[0], train_loss, train_acc, test_clean_loss, test_clean_acc, test_corrupt_loss, test_corrupt_acc))

        torch.save({
            'epoch': epoch + 1,
            'arch': cls_arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir))


def train(loader: DataLoader, model: torch.nn.Module, model_t, criterion, optimizer: Optimizer, epoch: int, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img, params = inputs
        img, params, targets = img.cuda(), params.cuda(), targets.cuda()    #img batch*3*32*32
        if model_t is not None:
            with torch.no_grad():
                params = sample_noise([params.shape[0],param_size], args.noise_sd, args.noise_dst).cuda()
                add_noise = sample_noise(img.shape, args.add_noise, args.noise_dst).cuda()
                if args.parallel:
                    img = model_t.module.sample_noise(img, params,add_noise)
                else:
                    img = model_t.sample_noise(img, params, add_noise)

        # augment inputs with noise
        # inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
        # compute output
        outputs = model(img)
        loss = criterion(outputs, targets.squeeze())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(acc1.item(), img.size(0))
        top5.update(acc5.item(), img.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

            if args.use_tb:
                print_iter = int(i/args.print_freq+epoch*len(loader)/args.print_freq)
                # print(print_iter)
                writer.add_scalar('training_loss,',losses.val, print_iter)
                writer.add_scalar('training_acc', top1.val, print_iter)

    return (losses.avg, top1.avg)


def test(loader: DataLoader, model: torch.nn.Module, criterion,  epoch , writer, corrupt=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()
    corrupt_marker = 'noisy' if corrupt else 'clean'

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            img, params = inputs
            img, params, targets= img.cuda(), params.cuda(), targets.cuda()


            # augment inputs with noise
            # inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            outputs = model(img)
            loss = criterion(outputs, targets.squeeze())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(acc1.item(), img.size(0))
            top5.update(acc5.item(), img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(corrupt_marker + ': [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        if args.use_tb:

            # writer.add_scalar('training_loss,', losses, print_iter)
            writer.add_scalar('test_acc'+'_'+corrupt_marker, top1.avg, epoch)

        return (losses.avg, top1.avg)


if __name__ == "__main__":
    main()
