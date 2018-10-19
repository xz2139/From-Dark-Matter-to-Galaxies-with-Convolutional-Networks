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
import torch.nn.functional as F

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
from Models import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def parse_args():
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument('--mini', type=int, default=0,
                        help='whether to use mini dataset.')
    parser.add_argument('--medium', type=int, default=0,
                        help='whether to use medium dataset.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--model_idx', type=int, default=0,
                        help='0:Unet, 1:baseline')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    return parser.parse_args()




def initial_loss(train_loader, val_loader, model, criterion):
    batch_time = AverageMeter()
    train_losses = AverageMeter()
    losses = AverageMeter()
    correct = 0
    correct_1 = 0
    total = 0
    # switch to train mode
    model.eval()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            # add a dimension, from (1, 32, 32, 32) to (1,1,32,32,32)
            input = input.unsqueeze(dim = 1).to(device).float()
            #target = target.unsqueeze(dim = 1).to(device).float()
            target = target.to(device).long()
            # compute output
            output = model(input)
            # print("target1: ", target.size())
            # print("output: ", output.size())
            loss = criterion(output, target)
            #print(loss)
            # measure accuracy and record loss
            train_losses.update(loss.item(), input.size(0))

        for i, (input, target) in enumerate(val_loader):
            # add a dimension, from (1, 32, 32, 32) to (1,1,32,32,32)

            input = input.unsqueeze(dim = 1).to(device).float()
            #target = target.unsqueeze(dim = 1).to(device).float()
            target = target.to(device).long()
            # compute output
            output = model(input)
            
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            outputs = F.softmax(output, dim=1)
            predicted = outputs.max(1, keepdim=True)[1]
            total += np.prod(target.shape)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
            correct_1 += torch.mul(predicted.eq(target.view_as(predicted)),(target >= 1)).sum().item()
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            #batch_time.update(time.time() - end)
            #end = time.time()

            
    #print('Inital Test: Loss {loss.avg:.4f} Accuracy {ac:.4f}\t'.format(loss=losses,ac=correct/total*100))
    print('Epoch Test: Loss {loss.avg:.4f} Accuracy {ac:.4f}  Recall {recall:.4f}\t'.format(loss=losses,ac=correct/total*100, recall = correct_1/total*100))

        
        # print('Training initial Loss {train_loss.avg:.4f}\t'
        # 	'Validation initial Loss {val_loss.avg:.4f}\t'.format(train_loss=train_losses, val_loss = val_losses))



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
        #target = target.unsqueeze(dim = 1).to(device).float()
        target = target.to(device).long()
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
    # batch_time = AverageMeter()
    # losses = AverageMeter()


    # # switch to evaluate mode
    
    # model.eval()
    # with torch.no_grad():
    #     end = time.time()
    #     for i, (input, target) in enumerate(val_loader):
    #         input = input.unsqueeze(dim = 1).to(device).float()
    #         #target = target.unsqueeze(dim = 1).to(device).float()
    #         target = target.to(device).to(device).long()
            
    #         # compute output
    #         output = model(input)
            
    #         loss = criterion(output, target)
    #         # measure accuracy and record loss
    #         losses.update(loss.item(), input.size(0))

    #         # measure elapsed time
    #         batch_time.update(time.time() - end)
    #         end = time.time()

            
    #print('Epoch Test: Loss {loss.avg:.4f}\t'.format(loss=losses))

    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode  
    correct = 0
    correct_1 = 0
    total = 0
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.unsqueeze(dim = 1).to(device).float()
#             target = target.unsqueeze(dim = 1).to(device).float()
            target = target.to(device).long()
            
            
            # compute output
            output = model(input)
            outputs = F.softmax(output, dim=1)
            predicted = outputs.max(1, keepdim=True)[1]
            total += np.prod(target.shape)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
            correct_1 += torch.mul(predicted.eq(target.view_as(predicted)),(target >= 1)).sum().item()
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            #batch_time.update(time.time() - end)
            #end = time.time()

            
    #print('Inital Test: Loss {loss.avg:.4f} Accuracy {ac:.4f}\t'.format(loss=losses,ac=correct/total*100))
    print('Epoch Test: Loss {loss.avg:.4f} Accuracy {ac:.4f}  Recall {recall:.4f}\t'.format(loss=losses,ac=correct/total*100, recall = correct_1/total*100))

def main():

    params = parse_args()
    print("arguments: %s" %(params))
    mini = params.mini
    medium = params.medium
    lr = params.lr
    model_idx = params.model_idx
    epochs = params.epochs
    batch_size = params.batch_size
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
    params = {'batch_size': batch_size,
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
    if model_idx == 0:
        model = SimpleUnet(dim).to(device)
    elif model_idx == 1:
        model = Baseline(dim, dim).to(device)
    #criterion = nn.MSELoss().to(device) #yueqiu
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr,
                                    weight_decay=args.weight_decay)
    initial_loss(training_generator, validation_generator, model, criterion)

    for epoch in range(epochs):
        adjust_learning_rate(lr, optimizer, epoch)
        train(training_generator, model, criterion, optimizer, epoch)
        # evaluate on validation set
        validate(validation_generator, model, criterion)



if __name__ == '__main__':
    main()
