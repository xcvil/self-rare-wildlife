import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

import models.builder as builder
import models.ResNet as ResNet

import json

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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

# knn monitor
parser.add_argument('--knn-k', default=1, type=int, help='k in kNN monitor')
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

    ngpus_per_node = torch.cuda.device_count()
    print('there is/are {} GPUs per nodes'.format(ngpus_per_node))
    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = builder.MoCo(ResNet.__dict__[args.arch],
                             args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, two_branch=True, normlinear=True)
    model.cuda()
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    memdir = os.path.join(args.knn_data, 'train')
    testdir = os.path.join(args.knn_data, 'test')
    normalize = transforms.Normalize(mean=[0.34098161014906836, 0.47044207777359126, 0.5797972380147923],
                                     std=[0.10761384273454896, 0.11021859651496183, 0.12975552642180524])

    test_aug = [
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        normalize
    ]
    memory_dataset = datasets.ImageFolder(memdir, transforms.Compose(test_aug))
    memory_loader = torch.utils.data.DataLoader(memory_dataset, batch_size=args.batch_size, shuffle=False,
                                                num_workers=2, pin_memory=True)

    test_dataset = datasets.ImageFolder(testdir, transforms.Compose(test_aug))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2,
                                              pin_memory=True)

    instfe, labelfe = encode(test_loader, model.encoder_q, args)
    np.save(args.save_dir + 'inst_feat.npy', instfe)
    np.save(args.save_dir + 'label.npy', labelfe)

    # logging
    results = {'knn-k': [], 'test_acc@1': []}

    for i in range(0,600):
        test_acc_1 = test(model.encoder_q, memory_loader, test_loader, i, args)
        results['knn-k'].append(args.knn_k)
        results['test_acc@1'].append(test_acc_1)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, i + 2))
        data_frame.to_csv(args.save_dir + 'log.csv', index_label='epoch')
        args.knn_k += 1


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
                'Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, 600, total_top1 / total_num * 100))
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


def encode(train_loader, model, args):
    # switch to train mode
    model.eval()
    num_data = len(train_loader)
    inst_feat = np.zeros((num_data, 128)) # store the features
    label_list = np.zeros((num_data,))

    for i, (images, labels) in tqdm(enumerate(train_loader), desc='Feature extracting'):
        images = images.cuda(non_blocking=True)
        # compute output
        feature = model(images)
        feature = F.normalize(feature[0], dim=1)
        instDis = feature.cpu().data.numpy()
        inst_feat[i] = instDis
        label_list[i] = labels
        # print('encoding image {} is finished'.format(str(i)))
    return inst_feat, label_list


if __name__ == '__main__':
    main()
