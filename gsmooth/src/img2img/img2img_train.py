
import sys
sys.path.append('../..')

import yaml
import random
import numpy as np
import argparse
import os
import torch
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss, SmoothL1Loss
from torch.utils.data import DataLoader
# from ..datasets import get_dataset, DATASETS
from src.img2img.img2img_data import img2img_dataset, DATASETS
# from ..architectures import ARCHITECTURES, get_architecture
from src.img2img.img2img_models import img2img_get_model, get_size

from torch.optim import SGD, Optimizer, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR
import time
import datetime
import torch.nn.functional as F
from src.train_utils import AverageMeter, accuracy, init_logfile, log
from torch.utils.tensorboard import SummaryWriter
from src.train_utils import log_codes


'''

Code for training an image-to-image transformation network


'''


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str,
                    default='cifar10', choices=["imagenet", "cifar10","cifar100", "mnist","tiny_imagenet"])

parser.add_argument('--outdir', type=str,
                    default='./logs/train_models',
                    help='folder to save model and training log')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')

parser.add_argument('--scale', type=str, default=1,
                    help='super resolution scale')



parser.add_argument('--arch', type=str,
                    default=["edsr","unet","runet"][1])
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')


#EDSR configuration
parser.add_argument('--kernel_size',type=int, default=3)
parser.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser.add_argument('--n_feats',type=int, default=64)



parser.add_argument('--corrupt',type=str,default=['none','gaussian_blur','motion_blur','rotational_blur','zoom_blur','defocus_blur','rotate','translate','contrast','pixelate','jpeg','scaling'][0],
                    help=' The corruption type for training')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--parallel',default=True, type=int,help='Use parallel training')
parser.add_argument('--use_tb', type=int,
                    default=False,
                    help='Use tensorboard')


parser.add_argument('--config_file',type=str, default=None)
parser.add_argument('--save_model',type=bool,default=True)

args = parser.parse_args()
if args.config_file is not None:
        data = yaml.load(open(args.config_file,'r'))
        for key, value in data.items():
            args.__dict__[key] = value


comment = args.arch+'_'+args.corrupt+'_m'
def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # seed = 20210
    # os.environ['PYTHONHASHSEED'] = '0'
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    logfilename = os.path.join(args.outdir, 'log') + time.strftime('_%m%d_%H_%M') + args.corrupt + '_' + comment +'.txt'
    args.outdir = args.outdir + time.strftime('/checkpoint_%m%d_%H_%M') + args.corrupt + '_' + comment +'.pth.tar'

    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain mae\ttestloss\ttest mae")

    if args.use_tb:

        writer = SummaryWriter(log_dir='./logs' + time.strftime('/%m%d_%H_%M') + args.corrupt + '_' + comment,
                               comment='trainning logs')

    else:
        writer = None

    funlog = [
        ('./corruption.py', 'Corruption'),
        ('./unet.py','UNet'),

    ]
    log(logfilename, str(args), prt=True)

    log(logfilename,log_codes(funlog),prt=False)



    train_dataset = img2img_dataset(args,args.dataset, 'train',args.corrupt)
    test_dataset = img2img_dataset(args, args.dataset, 'test', args.corrupt)
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=512,
                             num_workers=args.workers, pin_memory=pin_memory)

    #get dataset size
    img_size, param_size = get_size(args)


    model = img2img_get_model(args, img_size, param_size, parallel=args.parallel)
    # print(model)
    log(logfilename, str(model),prt=True)



    # criterion = CrossEntropyLoss().cuda()
    # criterion = MSELoss().cuda()
    criterion = L1Loss().cuda()
    # optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)

    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    print(criterion,optimizer)
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        before = time.time()
        train_loss, train_mae = train(train_loader, model, criterion, optimizer, epoch, args.noise_sd, writer)
        test_loss, test_mae = test(test_loader, model, criterion, args.noise_sd,epoch, writer)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_lr()[0], train_loss, train_mae, test_loss, test_mae))

        if args.save_model:
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir))


def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mses = AverageMeter()
    maes = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        img, params = inputs
        img, params, targets = img.cuda(), params.cuda(), targets.cuda()    #img batch*3*32*32
        # params = params.cuda()
        # targets = targets.cuda()

        # augment inputs with noise
        # inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

        # compute output

        outputs = model(img, params)
        loss = criterion(outputs, targets)
        # measure accuracy and record loss
        mse = F.mse_loss(outputs, targets)
        mae = F.l1_loss(outputs, targets)

        losses.update(loss.item(), img.size(0))
        mses.update(mse.item(), img.size(0))
        maes.update(mae.item(), img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MSE {mse.val:.4f} ({mse.avg:3f})\t'
                  'MAE {mae.val:.3f} ({mae.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses,mse = mses, mae = maes))

            if args.use_tb:
                print_iter = int(i/args.print_freq+epoch*len(loader)/args.print_freq)
                # print(print_iter)
                writer.add_scalar('training_loss,',losses.val, print_iter)
                writer.add_scalar('training_mae', maes.val, print_iter)

    return (losses.avg, maes.avg)


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float, epoch , writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    maes = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            img, params = inputs
            img, params, targets = img.cuda(), params.cuda(), targets.cuda()

            # augment inputs with noise
            # inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            outputs = model(img, params)
            loss = criterion(outputs, targets)


            # measure accuracy and record loss
            mae = F.l1_loss(outputs, targets)
            losses.update(loss.item(), img.size(0))
            maes.update(mae.item(), img.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE@1 {mae.val:.3f} ({mae.avg:.3f})'
                      .format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae=maes))

        if args.use_tb:

            # writer.add_scalar('testing_loss,', losses.avg, epoch)
            writer.add_scalar('test_mae', maes.avg, epoch)

        return (losses.avg, maes.avg)


if __name__ == "__main__":
    main()
    # print('end?:',input())