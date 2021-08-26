#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import models.loader as loader
import models.builder as builder
import models.ResNet as ResNet

import datetime
import json

import torchvision.models as models
from spectral_clustering import spectral_clustering, pairwise_cosine_similarity, KMeans

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default=None, type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--save-dir', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-xzheng', action='store_true',
                    help='use color+geo data augmentation')
parser.add_argument('--aug-color', action='store_true',
                    help='use only color data augmentation')
parser.add_argument('--aug-geo', action='store_true',
                    help='use geo data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# options for mix precision training
parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')

# options for CLD's hyper-parameter
parser.add_argument('--cld-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--Lambda', default=1.0, type=float,
                    help='lambda of branch two')
parser.add_argument('--normlinear', action='store_true',
                    help='use normlinear layer for projection head')
parser.add_argument('--clusters', default=100, type=int,
                    help='num of clusters for clustering')
parser.add_argument('--k-eigen', default=100, type=int,
                    help='num of eigenvectors for k-way normalized cuts')
parser.add_argument('--num-iters', default=10, type=int,
                    help='num of iters for clustering')
parser.add_argument('--use-kmeans', action='store_true', 
                    help='Whether use two randomly processed images')

# knn monitor
parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')
parser.add_argument('--knn-data', default='', type=str, metavar='PATH',
                    help='path to dataset of KNN')


def main():
    global args
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    path = os.path.join(args.save_dir, "config.json")
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print("Full config saved to {}".format(path))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print('Using distributed training')

    ngpus_per_node = torch.cuda.device_count()
    print('there is/are {} GPUs per nodes'.format(ngpus_per_node))
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = builder.MoCo(ResNet.__dict__[args.arch],
                         args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, two_branch=True, normlinear=args.normlinear)
    print(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.amp_opt_level != "O0":
        if amp is None:
            print("apex is not installed but amp_opt_level is set to {args.amp_opt_level}, ignoring.\n"
                           "you should install apex from https://github.com/NVIDIA/apex#quick-start first")
            args.amp_opt_level = "O0"
        else:
            model, optimizer = amp.initialize(model.cuda(), optimizer, opt_level=args.amp_opt_level)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            # optimizer = checkpoint['optimizer']
            if args.amp_opt_level != "O0" and checkpoint['args'].amp_opt_level != "O0":
                amp.load_state_dict(checkpoint['amp'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = args.data
    memdir = os.path.join(args.knn_data, 'train')
    testdir = os.path.join(args.knn_data, 'test')
    normalize = transforms.Normalize(mean=[0.34098161014906836, 0.47044207777359126, 0.5797972380147923],
                                     std=[0.10761384273454896, 0.11021859651496183, 0.12975552642180524])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomCrop(224),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    base_augmentation = [
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.RandomRotation([90, 90]),
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomRotation([180, 180]),
        ], p=0.5)
    ]

    key_augmentation = [
        transforms.RandomCrop(224),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    geo_augmentation = [
        transforms.RandomCrop(224),
        transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation([90, 90]),
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomRotation([180, 180]),
        ], p=0.5),
        transforms.ToTensor(),
        normalize
    ]

    test_aug = [
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize
    ]

    if args.aug_xzheng:
        train_dataset = datasets.ImageFolder(
            traindir,
            loader.ThreeCropsTransform(transforms.Compose(base_augmentation),
                                       transforms.Compose(geo_augmentation),
                                       transforms.Compose(key_augmentation),
                                       transforms.Compose(augmentation)))
        print('Using Geo and Color transformation')
    elif args.aug_color:
        train_dataset = datasets.ImageFolder(
            traindir,
            loader.ThreeCropsTransform(transforms.Compose(base_augmentation),
                                       transforms.Compose(augmentation),
                                       transforms.Compose(key_augmentation),
                                       transforms.Compose(augmentation)))
        print('Using only MoCo v2 augmentation')
    else:
        raise NotImplementedError("I haven't finished it! Someone helps me!!")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    memory_dataset = datasets.ImageFolder(memdir, transforms.Compose(test_aug))
    memory_loader = torch.utils.data.DataLoader(memory_dataset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=2, pin_memory=True)

    test_dataset = datasets.ImageFolder(testdir, transforms.Compose(test_aug))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)

    # logging
    results = {'train_loss': [], 'test_acc@1': []}

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, args)
        results['train_loss'].append(train_loss)
        test_acc_1 = test(model.module.encoder_q, memory_loader, test_loader, epoch, args)
        results['test_acc@1'].append(test_acc_1)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(args.start_epoch + 1, epoch + 2))
        data_frame.to_csv(args.save_dir + 'log.csv', index_label='epoch')

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer,
                'args': args,
            }
            if args.amp_opt_level != 'O0':
                state['amp'] = amp.state_dict()
            save_checkpoint(state, is_best=False, save_dir=args.save_dir, \
                filename='checkpoint_{:04d}.pth.tar'.format(epoch), epoch=epoch)

def grouping(features_groupDis1, features_groupDis2, T, args):
    # print(features_groupDis1.size())
    criterion = nn.CrossEntropyLoss().cuda()
    # K-way normalized cuts or k-Means. Default: k-Means
    if args.use_kmeans:
        cluster_label1, centroids1 = KMeans(features_groupDis1, K=args.clusters, Niters=args.num_iters)
        cluster_label2, centroids2 = KMeans(features_groupDis2, K=args.clusters, Niters=args.num_iters)
    else:
        cluster_label1, centroids1 = spectral_clustering(features_groupDis1, K=args.k_eigen,
                    clusters=args.clusters, Niters=args.num_iters)
        cluster_label2, centroids2 = spectral_clustering(features_groupDis2, K=args.k_eigen,
                    clusters=args.clusters, Niters=args.num_iters)

    # group discriminative learning
    affnity1 = torch.mm(features_groupDis1, centroids2.t())
    CLD_loss = criterion(affnity1.div_(T), cluster_label2)

    affnity2 = torch.mm(features_groupDis2, centroids1.t())
    CLD_loss = (CLD_loss + criterion(affnity2.div_(T), cluster_label1))/2

    return CLD_loss

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    total_loss, total_num = 0.0, 0
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            images[2] = images[2].cuda(args.gpu, non_blocking=True)

        # compute output
        outputs = model(im_q=images[0], im_k=images[1], im_q2=images[2])
        logits1, labels1, logits2, labels2, eq1, ek1, eq2 = outputs

        loss = criterion(logits1, labels1)/2 + criterion(logits2, labels2)/2
        loss += args.Lambda*grouping(eq1, eq2, args.cld_t, args)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits1, labels1, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward()
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        total_num += train_loader.batch_size
        total_loss += loss.item() * train_loader.batch_size

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, args)

    return total_loss / total_num


# test using a knn monitor
def test(model, memory_data_loader, test_data_loader, epoch, args):
    model.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_num, feature_bank = 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = model(data.cuda(non_blocking=True))
            feature = F.normalize(feature[0], dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = model(data)
            feature = F.normalize(feature[0], dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description(
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

    return total_top1 / total_num * 100


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


def save_checkpoint(state, is_best, save_dir='output/imagenet/', filename='checkpoint.pth.tar', epoch=0):
    # torch.save(state, filename)
    if (epoch + 1) % 10 == 0:
        torch.save(state, os.path.join(save_dir,filename))
    torch.save(state, os.path.join(save_dir, "current.pth.tar"))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, args):
        time = str(datetime.datetime.now())
        prefix = time + ' lr: {:.4f}\t'.format(args.current_lr) + self.prefix
        entries = [prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        message = '\t'.join(entries)
        print(message)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    args.current_lr = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
