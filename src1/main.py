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
TRAIN_LOSS,VAL_LOSS, VAL_ACC, VAL_RECALL, VAL_PRECISION = [],[],[],[],[]
BEST_VAL_LOSS = 999999999
BEST_RECALL = 0
BEST_PRECISION = 0
BEST_F1SCORE = 0
BEST_ACC = 0
EPSILON = 1e-5
if not os.path.exists('pretrained'):
    os.makedirs('pretrained')

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
                        help='0:Unet, 1:baseline 2: Inception 3. R2Unet 4.two-phase model(classfication phase: one layer Conv, regression phase: R2Unet)\
                         5.two-phase model(classfication phase: R2Unet, regression phase: R2Unet) 6. R2Unet attention \
                         7. two-phase model(classfication phase: Inception, regression phase: R2Unet) 8. Incetion regression')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--loss_weight', type=float, default=20,
                        help='weight of the loss equals to normalized [x, loss_weight * x,loss_weight * x]')
    parser.add_argument('--target_cat', default='count',
                        help='the target cube we want to predict, count or mass')
    parser.add_argument('--target_class', type = int, default= 0,
                        help='0:classification 1:regression')
    parser.add_argument('--plot_label', default= '',
                        help='label for the filename of the plot. If left default, \
                        the plot_label will be \'_\' + target_class + \'_\' + target_cat. \
                        This label is for eliminating risk of overwriting previous plot')
    parser.add_argument('--load_model', type=int, default=0,
                        help='')
    parser.add_argument('--save_name', default='',
                        help='the name of the saved model file, default don\'t save')
    parser.add_argument('--conv1_out', type=int, default=6,
                        help='number of hidden units for the size = 1 kernel')
    parser.add_argument('--conv3_out', type=int, default=8,
                        help='number of hidden units for the size = 3 kernel')
    parser.add_argument('--conv5_out', type=int, default=10,
                        help='number of hidden units for the size = 5 kernel')
    parser.add_argument('--record_results', type=int, default=0,
                        help='whether to write the best results to all_results.txt')
    parser.add_argument('--yfloss_weight', type=float, default=0,
                        help='')
    parser.add_argument('--vel', type=int, default=0,
                        help='whether to include velocity to the input(input dim 1 if not, 4 if yes)')
    parser.add_argument('--normalize', type=int, default=0,
                        help='whether to normalize the input(dark matter density)')
    parser.add_argument('--C_model', default="",
                        help='classfication model name for the two-phase model')

    return parser.parse_args()


def initial_loss(train_loader, val_loader, model, criterion, target_class):
    #AverageMeter is a object that record the sum, avg, count and val of the target stats
    train_losses = AverageMeter()
    val_losses = AverageMeter()  
    correct = 0
    # ptotal = 0  #count of all positive predictions
    # tp = 0    #true positive
    total = 0 #total count of data
    TPRs = AverageMeter()
    FPRs = AverageMeter()
    # switch to train mode
    model.eval()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            # add a dimension, from (1, 32, 32, 32) to (1,1,32,32,32)
            input = input.to(device).float()
            if target_class == 0:
                target = target.to(device).long()
            elif target_class == 1:
                target = target.to(device).float()
            # compute output
            output = model(input)
            # print("target1: ", target.size())
            # print("output: ", output.size())
            loss = criterion(output, target)
            # measure accuracy and record loss
            train_losses.update(loss.item(), input.size(0))

        for i, (input, target) in enumerate(val_loader):
            # add a dimension, from (1, 32, 32, 32) to (1,1,32,32,32)

            input = input.to(device).float()
            if target_class == 0:
                target = target.to(device).long()
            elif target_class == 1:
                target = target.to(device).float()
            # compute output
            output = model(input)
            loss = criterion(output, target)
            # measure accuracy and record loss
            val_losses.update(loss.item(), input.size(0))
            if target_class == 0:
                outputs = F.softmax(output, dim=1)
                predicted = outputs.max(1)[1]
                total += np.prod(target.shape)
                correct += predicted.eq(target.view_as(predicted)).sum().item()
                #ptotal += (target.view_as(predicted) >= 1).sum().item()
                #tp += torch.mul(predicted.eq(target.view_as(predicted)),(target.view_as(predicted)>= 1)).sum().item()
                TPR, gp, FPR, gf = confusion_matrix_calc(predicted,target)
                TPRs.update(TPR,gp)
                FPRs.update(FPR,gf)            
            loss = criterion(output, target)
            # measure accuracy and record loss
            val_losses.update(loss.item(), input.size(0))  

    # recall =  tp/ptotal*100  #recall = true positive / count of all positive predictions  
    if target_class == 0:
        acc = correct/total*100
        recall = TPRs.avg * 100
        precision = TPRs.sum/(TPRs.sum + FPRs.sum + EPSILON) * 100  
        VAL_RECALL.append(recall)
        VAL_ACC.append(acc)
        VAL_PRECISION.append(precision)  

    TRAIN_LOSS.append(train_losses.avg)
    VAL_LOSS.append(val_losses.avg)
    if target_class == 0:
        print('Epoch Train Loss {train_losses.avg:.4f}, Test Loss {val_losses.avg:.4f},\
         Test Accuracy {acc:.4f},  Test Recall {recall:.4f}\t Precision {precision:.4f}\t'.format(train_losses = train_losses, \
            val_losses=val_losses,acc=acc, recall = recall, precision = precision))
    else:
        print('Epoch Train Loss {train_losses.avg:.4f}, Test Loss {val_losses.avg:.4f}'\
            .format(train_losses = train_losses, val_losses=val_losses))

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
        input = input.to(device).float()
        if target_class == 0:
            target = target.to(device).long()
        elif target_class == 1:
            target = target.to(device).float()
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
    print('Epoch {0} : Train: Loss {loss.avg:.4f}\t'.format(epoch, loss=losses))
    


def validate(val_loader, model, criterion, epoch, target_class, save_name):
    global BEST_VAL_LOSS
    global BEST_RECALL
    global BEST_PRECISION
    global BEST_F1SCORE
    global BEST_ACC
    batch_time = AverageMeter()
    val_losses = AverageMeter()
    TPRs = AverageMeter()
    FPRs = AverageMeter()  
    total = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device).float()
            if target_class == 0:
                target = target.to(device).long()
            elif target_class == 1:
                target = target.to(device).float()
            
            
            # compute output
            output = model(input)
            if target_class == 0:
                outputs = F.softmax(output, dim=1)
                predicted = outputs.max(1)[1]
                total += np.prod(target.shape)
                correct += predicted.eq(target.view_as(predicted)).sum().item()
                TPR, gp, FPR, gf = confusion_matrix_calc(predicted,target)
                TPRs.update(TPR,gp)
                FPRs.update(FPR,gf)
            loss = criterion(output, target)
            # measure accuracy and record loss
            val_losses.update(loss.item(), input.size(0))

    
    if target_class == 0:
        recall = TPRs.avg * 100
        precision = TPRs.sum/(TPRs.sum + FPRs.sum + EPSILON) * 100
        F1score = 2*((precision*recall)/(precision+recall+ EPSILON))
        acc = correct/total*100
        VAL_RECALL.append(recall)
        VAL_ACC.append(acc)
        VAL_PRECISION.append(precision)
        if val_losses.avg < BEST_VAL_LOSS:
            BEST_RECALL = recall
            BEST_PRECISION = precision
            BEST_F1SCORE = F1score
            BEST_ACC = acc
    
    if val_losses.avg < BEST_VAL_LOSS:
        if len(save_name) > 0:
            #torch.save(model, 'pretrained/' + str(save_name) + '.pt')
            torch.save(model.state_dict(), 'pretrained/' + str(save_name) + '.pth')
        BEST_VAL_LOSS = val_losses.avg
    VAL_LOSS.append(val_losses.avg)
    if target_class == 0:
        print('Epoch {0} :Val Loss {val_losses.avg:.4f},\
         Val Accuracy {acc:.4f},  Val Recall {recall:.4f}\t Precision {precision:.4f} F1 score  {F1score:.4f}\t'.format(epoch, \
            val_losses=val_losses,acc=acc, recall = recall, precision = precision, F1score = F1score))
    else:
        print('Epoch {0} : Val Loss {val_losses.avg:.4f}'\
            .format(epoch, val_losses=val_losses))


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    print("arguments: %s" %(args))
    mini = args.mini
    medium = args.medium
    medium1 = args.medium1
    lr = args.lr
    model_idx = args.model_idx
    epochs = args.epochs
    batch_size = args.batch_size
    loss_weight = args.loss_weight
    weight_decay = args.weight_decay
    print_freq = args.print_freq
    target_cat = args.target_cat
    target_class = args.target_class
    plot_label = args.plot_label
    load_model = args.load_model
    save_name = args.save_name
    conv1_out, conv3_out, conv5_out = args.conv1_out, args.conv3_out, args.conv5_out
    record_results = args.record_results
    yfloss_weight = torch.Tensor([args.yfloss_weight]).to(device)
    vel = args.vel
    normalize = args.normalize
    C_model = args.C_model


    #index for the cube, each tuple corresponds to a cude
    #test data
    if mini:
        train_data = [(832, 640, 224),(864, 640, 224)]
        val_data = [(832, 640, 224),(864, 640, 224)]
        test_data = [(832, 640, 224),(864, 640, 224)]
    else:
        if medium1:
            data_range = 130
            random_idx = 1
        elif medium:
            data_range = 512
            random_idx = 1
        else:
            data_range = 1024
            random_idx = 0
        
        pos=list(np.arange(0,data_range,32))
        ranges=list(product(pos,repeat=3))
        random.seed(7)
        if random_idx == 1:
            random.shuffle(ranges)
            train_data = ranges[:int(np.round(len(ranges)*0.6))]
            val_data=ranges[int(np.round(len(ranges)*0.6)):int(np.round(len(ranges)*0.8))]
            test_data = ranges[int(np.round(len(ranges)*0.8)):]
        else:
            train_data, val_data, test_data = [],[],[]

            for i in range(0,data_range,32):
                for j in range(0,data_range,32):
                    for k in range(0,data_range,32):
                        idx = (i,j,k)
                        if i <=416 and j<= 416:
                            val_data.append(idx)
                        elif i>=484 and j>= 448 and k>= 448:
                            test_data.append(idx)
                        else:
                            train_data.append(idx)
    # #build dataloader
    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers':20}

    training_set, validation_set = Dataset(train_data,cat = target_cat,reg = target_class, vel = vel), Dataset(val_data, cat = target_cat,reg = target_class, vel = vel, normalize = normalize)
    testing_set= Dataset(test_data, cat= target_cat, reg = target_class, vel = vel, normalize = normalize)
    training_generator = data.DataLoader(training_set, **params)
    validation_generator = data.DataLoader(validation_set, **params)
    testing_generator = data.DataLoader(testing_set, **params)

    # #set up device

    # #build model
    dim_out = 1
    if vel == 1:
        dim_in = 4
    else:
        dim_in = 1
        dim = 1 ## need to be changed later
    if model_idx == 0:
        model = SimpleUnet(dim, target_class).to(device)
    elif model_idx == 1:
        model = Baseline(dim, dim).to(device)
    elif model_idx == 2:
        model = Inception(dim_in, conv1_out, conv3_out, conv5_out).to(device)
    elif model_idx == 3:
        model = R2Unet(dim_in, dim_out, t = 3, reg = target_class).to(device)
    elif model_idx == 4:
        mask_model = one_layer_conv(dim,one_layer_outchannel = 8,kernel_size = 3,non_linearity = 'ReLU6', transformation = 'sqrt_root'
                                    , power = 0.25).to(device)
        #state_dict = torch.load('../trained_model/epoch_10_MSE.pth')
        state_dict = torch.load('./pretrained/' + C_model + '.pth')
        mask_model.load_state_dict('state_dict')
        pred_model = R2Unet(dim,dim,t=3,reg = target_class).to(device)
        model = two_phase_conv(mask_model,pred_model,thres = thres)
    elif model_idx == 5:
        mask_model = R2Unet(dim_in, dim_out, t = 3).to(device)
        state_dict = torch.load('./pretrained/' + C_model + '.pth')
        mask_model.load_state_dict(state_dict)
        pred_model = R2Unet(dim_in,dim_out,t=3,reg = target_class).to(device)
        model = two_phase_conv(mask_model,pred_model)
    elif model_idx == 6:
        model = R2Unet_atten(dim, dim, t = 3, reg = 0).to(device)
        
    elif model_idx == 7:
        mask_model = Inception(dim_in, conv1_out, conv3_out, conv5_out).to(device)
        state_dict = torch.load('./pretrained/' + C_model + '.pth')

        mask_model.load_state_dict(state_dict)
        pred_model = R2Unet(dim_in,dim_out,t=3,reg = target_class).to(device)
        model = two_phase_conv(mask_model,pred_model)
    elif model_idx == 8:
        model = Inception(dim, conv1_out, conv3_out, conv5_out, reg = target_class).to(device)
    else:
        print('model not exist')

    if load_model:
        model = torch.load('pretrained/mytraining.pt')
    #criterion = nn.MSELoss().to(device) #yueqiu
    #weight = torch.Tensor([0.99,0.05,0.05])
    if target_class == 0:
        criterion = nn.CrossEntropyLoss(weight = get_loss_weight(loss_weight, num_class = 2)).to(device)
        print('criterion classification')
        #criterion = yfloss(weight = get_loss_weight(loss_weight, num_class = 2).to(device), w = yfloss_weight, device = device)
    else:
        criterion = weighted_nn_loss(loss_weight)
        #criterion = nn.MSELoss() #yueqiu


    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    initial_loss(training_generator, validation_generator, model, criterion, target_class)

    for epoch in range(epochs):
        adjust_learning_rate(lr, optimizer, epoch)
        train(training_generator, model, criterion, optimizer, epoch, print_freq, target_class = target_class)
        #evaluate on validation set
        validate(validation_generator, model, criterion, epoch, target_class = target_class, save_name = save_name)
    if len(plot_label) == 0:
        plot_label = '_' + str(target_class) + '_' + str(model_idx) + '_'
    train_plot(TRAIN_LOSS,VAL_LOSS, VAL_ACC, VAL_RECALL, VAL_PRECISION, target_class, plot_label = plot_label)
    if target_class == 0:
        if record_results:
            args = parse_args()
            f= open("all_results","a+")
            f.write("arguments: %s" %(args) + '\n')
            f.write('Test Loss {BEST_VAL_LOSS:.5f},  Test Accuracy {BEST_ACC:.4f},  Test Recall {BEST_RECALL:.4f},  \
            Precision {BEST_PRECISION:.4f}   F1 score  {BEST_F1SCORE:.4f}\n'.format( \
                        BEST_VAL_LOSS=BEST_VAL_LOSS,BEST_ACC=BEST_ACC, BEST_RECALL = BEST_RECALL, BEST_PRECISION = BEST_PRECISION, BEST_F1SCORE =  BEST_F1SCORE))
            f.close() 


if __name__ == '__main__':
    main()
