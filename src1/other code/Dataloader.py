import os
import numpy as np
from torch.utils import data
import torch.nn as nn
import torch
import random
import numpy as np
from itertools import product
from torch.autograd import Variable
from dataset import Dataset

params = {'batch_size': 3,
          'shuffle': False,
          #'shuffle': True,
          'num_workers':20}
max_epochs = 100

training_set, validation_set = Dataset(train_data), Dataset(val_data)
testing_set= Dataset(test_data)
training_generator = data.DataLoader(training_set, **params)
validation_generator = data.DataLoader(validation_set, **params)
testing_generator = data.DataLoader(testing_set, **params)

for i, (dark,full) in enumerate(testing_generator):
    dark=dark
    full=full