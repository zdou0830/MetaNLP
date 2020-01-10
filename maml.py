import argparse
import numpy as np
import random
import torch
import math
from collections import defaultdict
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from model_maml import MTDNNModel
from tokenizer import BertTokenizer
from data_prep import convert_examples_to_features, MrpcProcessor, ColaProcessor, MnliProcessor, Sst2Processor, RteProcessor, WnliProcessor, QqpProcessor, QnliProcessor, StsbProcessor
from reload_tf_model import load_tf_weights, load_model_weights_from_tf_weights, return_model_weights_from_tf_weights
import os
import pickle
from torch.autograd import Variable

def maml():
    for i, task_id in enumerate(extra_ids):
        for update_step in range(args.num_update_steps+1):
            batch = load_next_batch(i, task_id)
                
            x, x_mask, x_token, y = tuple(t.to(device) for t in batch)

            if update_step == args.num_update_steps:
                if update_step == 0:
                    loss = model(params, task_id, x, x_mask, x_token, y) 
                    grads = torch.autograd.grad(loss, params, allow_unused=True)
                else:
                    loss = model(fast_params, task_id, x, x_mask, x_token, y) 
                    grads = torch.autograd.grad(loss, params, allow_unused=True)
                for param, grad in zip(params, grads):
                    if not param.requires_grad or grad is None:
                        continue
                    if param.grad is None:
                        param.grad = Variable(torch.zeros(grad.size()).cuda())
                    param.grad.data.add_(grad/args.num_sample_tasks)

            elif update_step == 0:
                loss = model(params, task_id, x, x_mask, x_token, y) 
                grad = torch.autograd.grad(loss, params, allow_unused=True)
                fast_params = list(map(lambda p: p[1] - args.inner_learning_rate * p[0] if p[0] is not None else p[1], zip(grad, params)))
            elif update_step < args.num_update_steps:
                loss = model(fast_params, task_id, x, x_mask, x_token, y) 
                grad = torch.autograd.grad(loss, fast_params, allow_unused=True)
                fast_params = list(map(lambda p: p[1] - args.inner_learning_rate * p[0] if p[0] is not None else p[1], zip(grad, fast_params)))

        if i % args.num_sample_tasks == (args.num_sample_tasks-1):
            optimizer.step()
            optimizer.zero_grad()
