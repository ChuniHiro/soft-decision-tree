from __future__ import print_function
import os
import argparse
import pickle
import torch
from torchvision import datasets, transforms

from model import SoftDecisionTree

from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
from statistics import mean, stdev
import os
import sys
import pandas as pd
import time
from tqdm import tqdm

from scipy import spatial

from collections import Counter

from joblib import Parallel, delayed
import multiprocessing

import math
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
import random
import pickle

from model import SoftDecisionTree
from torch.utils.data import Dataset, DataLoader


featname = "./model/feat_train_200K+test_dft=512.pkl"
feattraintmp,feattesttmp = pickle.load(open(featname ,'rb'))
print("feature loaded at ", featname)

yname = "./model/y_train_200K+test.pkl"
y_train_200K, y_test, = pickle.load(open(yname,'rb'))
print("label loaded at ", yname)

print(feattraintmp.shape, y_train_200K.shape)
print(feattesttmp.shape, y_test.shape)




class MyDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.int64)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
    

# create dataset
dataset_train = MyDataset(feattraintmp, y_train_200K)
dataset_test = MyDataset(feattesttmp, y_test)




# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
parser.add_argument('--modelname', type=str,
                    help='model name for saving')
parser.add_argument('--batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--input-dim', type=int, default=1024, metavar='N',
                    help='input dimension size(default: 28 * 28)')
parser.add_argument('--output-dim', type=int, default=10, metavar='N',
                    help='output dimension size(default: 10)')
parser.add_argument('--max-depth', type=int, default=6, metavar='N',
                    help='maximum depth of tree(default: 8)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 40)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lmbda', type=float, default=0.1, metavar='LR',
                    help='temperature rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden-size', nargs="+", type=int, default= [512, 256],
                    help='hhidden layer for MLP')
parser.add_argument('--linear', type=bool, default=False, metavar='N',
                    help='using linear or Relu for MLP')
parser.add_argument('--loadckpt', type=str, default="",
                    help='if using previous weights')
parser.add_argument('--device', type=str, default="cuda:0",
                    help='if using cuda')
# print(parser)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# args.batch_size = 10000
# args.max_depth = 6
# args.hidden_size = [512, 256]
# args.linear = True
print(args)

train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

model = SoftDecisionTree(args)
if args.cuda:
    model.cuda()

if len(args.loadckpt) > 0:
    
    with open(args.loadckpt, 'rb') as model_file:
        
        model = pickle.load( model_file)
        print("model loaded at:", args.loadckpt)
    
# print(model)
for epoch in range(1, args.epochs + 1):
    
    model.train_(train_loader, epoch)
    model.test_(test_loader, epoch)