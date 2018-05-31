# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn

import data
import model
import os
import os.path as osp
import numpy as np
from torch.nn.utils import clip_grad_norm

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used')
# Hyper-parameters

parser.add_argument('--nembed', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=1024,
                    help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of LSTM layers')
parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='number of LSTM layers')


args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
data_loader = data.Corpus("./data/ptb", args.batch_size, args.max_sql)

nvocab = len(data_loader.word_id)    
# WRITE CODE HERE witnin two '#' bar
########################################
# Build LMModel model (bulid your language model here)
model = model.LMModel(nvocab, args.nembed, args.nhidden, args.nlayers)
########################################

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# WRITE CODE HERE witnin two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.
def evaluate(data_source):
    pass
########################################


# WRITE CODE HERE witnin two '#' bar
########################################
# Train Function
def train():
    # Set initial hidden and cell states
    states = (torch.zeros(args.nlayers, args.batch_size, args.nhidden),
              torch.zeros(args.nlayers, args.atch_size, args.nhidden))
    data, target, end_flag = data_loader.get_batch()
    i = 0
    while not end_flag:
    # Get mini-batch inputs and targets
        inputs = data
        targets = target
        
        # Forward pass
        states = detach(states)
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
        
        # Backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), 0.5)
        optimizer.step()

        step = (i+1) // args.max_sql
        if step % 100 == 0:
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                  .format(epoch+1, args.epochs, step, data_loader.train_batch_num, loss.item(), np.exp(loss.item())))
        data, target, end_flag = data_loader.get_batch()
        i = i + 1
########################################

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 

# Loop over epochs.
for epoch in range(1, args.epochs+1):
    train()
    # evaluate()

