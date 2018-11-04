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

from train_f import *
from Dataset import Dataset
from Models import *
from args import args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# the following four variables are global variables that record the statistics 
# for each epoch so that the plot can be produced
TRAIN_LOSS,VAL_LOSS, VAL_ACC, VAL_RECALL = [],[],[],[]



def parse_args():
    parser = argparse.ArgumentParser(description="main.py")

    
    parser.add_argument('--mini', type=int, default=0,
                        help='whether to use mini dataset.')
    parser.add_argument('--medium1', type=int, default=0,
                        help='whether to use medium dataset.(2% of the data)')
    parser.add_argument('--medium', type=int, default=0,
                        help='whether to use medium dataset.(12.5% of the data)')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='')
    parser.add_argument('--print_freq', type=int, default=400,
                        help='')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--model_idx', type=int, default=0,
                        help='0:Unet, 1:baseline')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--loss_weight', type=int, default=300,
                        help='weight of the loss equals to normalized [x, loss_weight * x,loss_weight * x]')
    parser.add_argument('--target_cat', default='count',
                        help='the target cube we want to predict, count or mass')
    parser.add_argument('--target_class', type = int, default= 0,
                        help='0:classification 1:regression')
    return parser.parse_args()




def initial_loss(train_loader, val_loader, model, criterion):
    #AverageMeter is a object that record the sum, avg, count and val of the target stats
    train_losses = AverageMeter()
    val_losses = AverageMeter()  
    correct = 0
    ptotal = 0  #count of all positive predictions
    tp = 0    #true positive
    total = 0 #total count of data
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
            val_losses.update(loss.item(), input.size(0))
            outputs = F.softmax(output, dim=1)
            predicted = outputs.max(1, keepdim=True)[1]
            total += np.prod(target.shape)
            correct += predicted.eq(target.view_as(predicted)).sum().item()
            ptotal += (target.view_as(predicted) >= 1).sum().item()
            tp += torch.mul(predicted.eq(target.view_as(predicted)),(target.view_as(predicted)>= 1)).sum().item()
            # print('predicted.size() = ',  predicted.size())
            # print('target.size() = ',  target.size() )
            # print(torch.mul(predicted.view_as(target).eq(target),(target >= 1)).size())
            # print('xinyue tp = ', (torch.mul(predicted,target.view_as(predicted))).sum().item())
            # print('xinyue ptotal = ', (target.view_as(predicted) >= 1).sum().item())
            # equal = predicted.eq(target).data
            # multi = torch.mul(predicted.eq(target),(target >= 1)).data
            # print('equal = ', equal[0:10, 0:20, 0])
            # print('multi = ', multi[0:10, 0:20, 0])
            loss = criterion(output, target)
            # measure accuracy and record loss
            val_losses.update(loss.item(), input.size(0))  
    # print('tp = ', tp)
    # print('ptotal = ', ptotal)
    # print('total = ', total)
    # print('correct = ', correct)
    recall =  tp/ptotal*100  #recall = true positive / count of all positive predictions  
    acc = correct/total*100
    TRAIN_LOSS.append(train_losses.avg)
    VAL_LOSS.append(val_losses.avg)
    VAL_RECALL.append(recall)
    VAL_ACC.append(acc)
    print('Epoch Train Loss {train_losses.avg:.4f}, Test Loss {val_losses.avg:.4f},\
     Test Accuracy {acc:.4f},  Test Recall {recall:.4f}\t'.format(train_losses = train_losses, \
        val_losses=val_losses,acc=acc, recall = recall))



def train(train_loader, model, criterion, optimizer, epoch, print_freq, target_class):


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

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    TRAIN_LOSS.append(losses.avg)
    print('Epoch Train: Loss {loss.avg:.4f}\t'.format(loss=losses))

def validate(val_loader, model, criterion, target_class):


    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to evaluate mode  
    correct = 0
    tp = 0  #true positive
    total = 0 #total count of data
    ptotal = 0 #count of all positive predictions
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
            ptotal += (target.view_as(predicted) >= 1).sum().item()
            tp += torch.mul(predicted.eq(target.view_as(predicted)) ,(target.view_as(predicted)>= 1)).sum().item()
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

    recall =  tp/ptotal*100  #recall = true positive / count of all positive predictions  
    acc = correct/total*100
    VAL_LOSS.append(losses.avg)
    VAL_RECALL.append(recall)
    VAL_ACC.append(acc)

    #print('Inital Test: Loss {loss.avg:.4f} Accuracy {ac:.4f}\t'.format(loss=losses,ac=correct/total*100))
    print('Test: Loss {loss.avg:.4f} Accuracy {ac:.4f}  Recall {recall:.4f}\t'.format(loss=losses,ac=correct/total*100, recall = tp/ptotal*100))

def main():

    params = parse_args()
    print("arguments: %s" %(params))
    mini = params.mini
    medium = params.medium
    medium1 = params.medium1
    lr = params.lr
    model_idx = params.model_idx
    epochs = params.epochs
    batch_size = params.batch_size
    loss_weight = params.loss_weight
    weight_decay = params.weight_decay
    print_freq = params.print_freq
    target_cat = params.target_cat
    target_class = params.target_class
    #index for the cube, each tuple corresponds to a cude
    #test data
    if mini:
        train_data = [(832, 640, 224),(864, 640, 224)]
        val_data = [(832, 640, 224),(864, 640, 224)]
        test_data = [(832, 640, 224),(864, 640, 224)]
    else:
        if medium1:
            data_range = 512
        if medium:
            data_range = 512
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

    training_set, validation_set = Dataset(train_data, cat = target_cat), Dataset(val_data, cat = target_cat)
    testing_set= Dataset(test_data, cat= target_cat)
    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params)
    testing_generator = data.DataLoader(testing_set, **params)

    # #set up device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # #build model
    dim = 1
    if model_idx == 0:
        model = SimpleUnet(dim).to(device)
    elif model_idx == 1:
        model = Baseline(dim, dim).to(device)
    elif model_idx == 2:
        model = Inception(dim).to(device)
    else:
        print('model not exist')
    #criterion = nn.MSELoss().to(device) #yueqiu
    #weight = torch.Tensor([0.99,0.05,0.05])
    if target_class == 0:
        criterion = nn.CrossEntropyLoss(weight = get_loss_weight(loss_weight, num_class = 2)).to(device)
    else:
        criterion = nn.MSELoss().to(device) #yueqiu

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    initial_loss(training_generator, validation_generator, model, criterion)

    for epoch in range(epochs):
        adjust_learning_rate(lr, optimizer, epoch)
        train(training_generator, model, criterion, optimizer, epoch, print_freq, target_class = target_class)
        # evaluate on validation set
        #validate(validation_generator, model, criterion, target_class = target_class)
    #train_plot(TRAIN_LOSS,VAL_LOSS, VAL_ACC, VAL_RECALL)


if __name__ == '__main__':
    main()
