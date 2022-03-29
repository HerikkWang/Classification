from asyncore import write
from torch._C import device
from ResNet20_raw import *
from CubeNets import *
from ResNet_whconv import *
from model_trial import *
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from options import parser
from utils.logging import Get_logger
# import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from one_net import *
from mlp_model import *

def main():
    global args, best_prec1
    best_prec1 = 0
    args = parser.parse_args()
    logger = Get_logger(os.path.join("./log", args.lp))
    # logger = Get_logger("test.log")
    torch.manual_seed(args.seed)
    logger.info("=====Parameter Setting=====")
    logger.info("Use model: " + args.model_name)
    logger.info("Use gpu: " + args.gpu)
    logger.info("Learning rate: " + "{:.5e}".format(args.lr))
    logger.info("Weight decay: " + "{:.5e}".format(args.weight_decay))
    logger.info("Learning rate milestones: " + str(args.lr_milestones))
    logger.info("Batch size: " + str(args.batch_size))
    logger.info("Epochs: " + str(args.epochs))
    logger.info("===========================")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    writer = SummaryWriter(os.path.join("runs", args.experiment_name))


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_dict = {'ResNet20_A': ResNet20_A, 
                #   'ResNet20_A_twist': ResNet20_A_twist,
                #   'ResNet20_A_twist_v1': ResNet20_A_twist_v1, 
                #   "ResNet20_A_twist_B12": ResNet20_A_twist_B12,
                #   "ResNet20_B_twist_B12": ResNet20_B_twist_B12, 
                #   "ResNet20_OptB_twist_B1": ResNet20_OptB_twist_B1,
                #   "ResNet20_B_twist_B1": ResNet20_B_twist_B1,
                #   "ResNet20_B_twist_B1_nodownsample": ResNet20_B_twist_B1_nodownsample,
                #   'ResNet20_B': ResNet20_B,
                #   'Cube_net': Cube_net,
                #   'Cube_net_conv1': Cube_net_conv1,
                #   "Cube_net_16_16": Cube_net_16_16,
                #   "Cube_net_32_16": Cube_net_32_16,
                #   "Cube_net_32_16_bn": Cube_net_32_16_bn,
                #   "Cube_net_32_v2": Cube_net_32_v2,
                #   "Cube_net_32_ordinary": Cube_net_32_ordinary,
                #   "Cube_net_32_bn_twist": Cube_net_32_bn_twist,
                  "Cube_net_cn": Cube_net_cn,
                  "ResNet_with_cube_block": ResNet_with_cube_block,
                  "ResNet20_B_plug": ResNet20_B_plug,
                  "ResNet20_B_bkplug": ResNet20_B_bkplug,
                  "ResNet20_B_bkplug_2": ResNet20_B_bkplug_2,
                  "ResWHNet": ResWHNet,
                  "model_trial_v0": model_trial_v0, 
                  "ResNet56_modified": ResNet56_modified,
                  "ResNet42_modified": ResNet42_modified, 
                  "test_model_1": test_model_1,
                  "ResNet20_B_improved": ResNet20_B_improved,
                  "model_trial_v1": model_trial_v1,
                  "softmax_conv_model": softmax_conv_model,
                  "one_net": one_net,
                  "oneconv_net": oneconv_net,
                  "ConvMixerShuffleNet": ConvMixerShuffleNet,
                  "ConvMixerShuffleNet2": ConvMixerShuffleNet2,
                  "NormNet2": NormNet2,
                  "NoPaddingNet": NoPaddingNet,
                  "SparseMLP": SparseMLP,

    }
    model = model_dict[args.model_name](num_classes=args.num_classes)
    # model = model_dict[args.model_name](args)
    model.to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_milestones, last_epoch=args.start_epoch - 1)

    # if args.arch in ['resnet1202', 'resnet110']:
    #     # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
    #     # then switch back. In this setup it will correspond for first epoch.
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        logger.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, logger)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, device, logger)
        writer.add_scalar("validate accuracy", prec1 / 100, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1, epoch)

        if args.if_save:
            if epoch > 0 and (epoch + 1) % args.save_every == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'model.th'))


def train(train_loader, model, criterion, optimizer, epoch, logger):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        # print(output.size())
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, device, logger):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    logger.info(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
