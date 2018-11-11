import torch
import numpy as np
import os
from args import args
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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

def adjust_learning_rate(lr0, optimizer, epoch):

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr0 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
#output a tensor equals to normalized [x, loss_weight * x,loss_weight * x]
def get_loss_weight(loss_weight, num_class):
    piece = 1/((num_class - 1) * loss_weight + 1)
    a = [1]
    a.extend([loss_weight] * (num_class - 1))
    return (torch.from_numpy(piece * np.array(a))).float()

def weighted_nn_loss(weight_ratio):
    def weighted(X,Y):
        base_loss = F.mse_loss(X,Y,reduction = 'sum')
        index = Y > 0
        plus_loss = F.mse_loss(X[index],Y[index], reduction = 'sum') if index.any() > 0 else 0
        total_loss = base_loss + (weight_ratio -1) * plus_loss
        return total_loss/X.shape[0]
    return weighted
    
def confusion_matrix_calc(pred,Y):
    Y_index = Y > 0
    TP = torch.sum(pred[Y_index] > 0).item()   #recall calculation
    FP = torch.sum(pred[~Y_index] > 0).item() #how many false positive
    ground_P = Y[Y_index].numel()
    ground_F = Y.numel() - ground_P
    return TP/ground_P, ground_P, FP/ground_F, ground_F
    
def train_plot(train_loss, val_loss, val_acc, val_recall, val_precision, target_class, plot_label = ''):
    if not os.path.exists('fig'):
        os.makedirs('fig')
    fig_dir = './fig/' + plot_label
    if target_class == 0:
        plt.figure()
        plt.plot(val_recall,label='Validation Recall')
        plt.plot(val_acc,label='Validation Accuracy')
        plt.plot(val_precision,label='Validation Precision')
        plt.legend()
        plt.savefig(fig_dir + 'recall+acc+precision')
    plt.figure()
    plt.plot(train_loss,label='Training Loss')
    plt.plot(val_loss,label='Validation Loss')
    plt.legend()
    plt.savefig(fig_dir + 'loss')

def blob_loss(predicted, kernel_size = 3):
    predict = predicted
    num_obs = predicted.size(0)
    length = predicted.size(1)
    cube_sums = []
    for i0 in range(num_obs):
        for i in range(0,length - kernel_size + 1):
            for j in range(0,length - kernel_size + 1):
                for k in range(0,length - kernel_size + 1):
                    cube_sum = torch.sum(predicted[i0,i:i + kernel_size, j:j + kernel_size,k:k + kernel_size])
                    cube_sum_1 = 0 if cube_sum < 2 else (cube_sum.item() - 1) **2
                    cube_sums.append(cube_sum_1)
    return torch.sum(torch.Tensor(cube_sums)) / torch.prod(torch.Tensor(list(predicted.size())))



def yqloss(weight, w):
    def yqloss_(pred, target, kernel_size = 3):
        criterion = nn.CrossEntropyLoss(weight = weight)
        loss_nn = criterion(pred, target)
        loss_blob = blob_loss(predicted, kernel_size = kernel_size)
        loss = loss_nn + torch.tensor(w * loss_blob)
        return loss_nn
    return yqloss_
