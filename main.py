import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models

from torch.utils import data
import random
import numpy as np
from itertools import product
import argparse

from args import args
from train_f import *
from Dataset import Dataset
from Models import SimpleUnet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def parse_args():
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument('--mini', type=int, default=0,
                        help='whether to use mini dataset.')
    parser.add_argument('--medium', type=int, default=0,
                        help='whether to use medium dataset.')
    
    return parser.parse_args()


print("arguments: %s" %(args))

def initial_loss(train_loader, val_loader, model, criterion):
    batch_time = AverageMeter()
    train_losses = AverageMeter()
    val_losses = AverageMeter()

    # switch to train mode
    model.eval()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            # add a dimension, from (1, 32, 32, 32) to (1,1,32,32,32)
            input = input.unsqueeze(dim = 1).to(device).float()
            target = target.unsqueeze(dim = 1).to(device).float()
            # compute output
            output = model(input)
            
            loss = criterion(output, target)
            # measure accuracy and record loss
            train_losses.update(loss.item(), input.size(0))

        for i, (input, target) in enumerate(val_loader):
            # add a dimension, from (1, 32, 32, 32) to (1,1,32,32,32)
            input = input.unsqueeze(dim = 1).to(device).float()
            target = target.unsqueeze(dim = 1).to(device).float()
            # compute output
            output = model(input)
            
            loss = criterion(output, target)
            # measure accuracy and record loss
            val_losses.update(loss.item(), input.size(0))


        
        print('Training initial Loss {train_loss.avg:.4f}\t'
        	'Validation initial Loss {val_loss.avg:.4f}\t'.format(train_loss=train_losses, val_loss = val_losses))



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)



        # add a dimension, from (1, 32, 32, 32) to (1,1,32,32,32)
        input = input.unsqueeze(dim = 1).to(device).float()
        target = target.unsqueeze(dim = 1).to(device).float()
        # compute output
        output = model(input)
        #print(torch.nonzero(target).size())
        loss = criterion(output, target)
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))


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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    print('Epoch Train: Loss {loss.avg:.4f}\t'.format(loss=losses))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()


    # switch to evaluate mode
    
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.unsqueeze(dim = 1).to(device).float()
            target = target.unsqueeze(dim = 1).to(device).float()
            
            
            # compute output
            output = model(input)
            
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            
    print('Epoch Test: Loss {loss.avg:.4f}\t'.format(loss=losses))

def main():

    params = parse_args()
    mini = params.mini
    medium = params.medium
    #index for the cube, each tuple corresponds to a cude
    #test data
    if mini:
        train_data = [(832, 640, 224),(864, 640, 224)]
        val_data = [(832, 640, 224),(864, 640, 224)]
        test_data = [(832, 640, 224),(864, 640, 224)]
    else:
        if medium:
            data_range = 150
        else:
            data_range = 1024

        pos=list(np.arange(0,data_range,32))
        ranges=list(product(pos,repeat=3))
        random.shuffle(ranges)
        train_data = ranges[:int(np.round(len(ranges)*0.6))]
        val_data=ranges[int(np.round(len(ranges)*0.6)):int(np.round(len(ranges)*0.8))]
        test_data = ranges[int(np.round(len(ranges)*0.8)):]

    # #build dataloader
    params = {'batch_size': 16,
          'shuffle': True,
          'num_workers':20}

    training_set, validation_set = Dataset(train_data), Dataset(val_data)
    testing_set= Dataset(test_data)
    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params)
    testing_generator = data.DataLoader(testing_set, **params)

    # for i, (input, target) in enumerate(training_generator):
    #     print('input')
    #     print(input)
    #     print('target')
    #     print(target)

    # #set up device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # #build model
    dim = 1
    model = SimpleUnet(dim).to(device)
    criterion = nn.MSELoss().to(device) #yueqiu
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)
    initial_loss(training_generator, validation_generator, model, criterion)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(training_generator, model, criterion, optimizer, epoch)
        # evaluate on validation set
        validate(validation_generator, model, criterion)



if __name__ == '__main__':
    main()
