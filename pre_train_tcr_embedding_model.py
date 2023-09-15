import torch
import argparse
import math
import numpy as np
from torch import nn
import pandas as pd
import random
from embeding import *
from torch.nn.utils import clip_grad_norm_
from bert_tcr import BERT
from pytorchtools import EarlyStopping
from torch.utils.data import DataLoader,TensorDataset

parser = argparse.ArgumentParser(description='train the tcr embedding model')

# File dir
parser.add_argument('--input', type=str, default="./data/train_tcr.csv", help='a csv with column "cdr3"')
parser.add_argument('--model_dir', type=str, default="./output/tcr_model.pt", help='output dir of model')

# Hyperparameters
parser.add_argument('--n_layers', type=int, default=4, help='number of transformer encoder layers')
parser.add_argument('--d_model', type=int, default=256, help='number of embedding dimention')
parser.add_argument('--batchsize', type=int, default=1024, help='mini batchsize')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=100, help="Maximum number of train epoch")

# GPUs
parser.add_argument('--GPUs', type=int, default=2, help='num of GPUs used in this task')

args = parser.parse_args()

# BERT Parameters
maxlen = 30 # max tokens of TCR sequence
max_pred = 5 # max tokens of prediction
n_layers = args.n_layers  # num of transformer encoder layers
n_heads = 8   # num of muti self-attention heads 
d_model = args.d_model   # embedding dimention
d_ff = d_model*4 # 4*d_model, FeedForward dimension
d_k = d_v = d_model*2  # dimension of K(=Q), V
mask_P = 0.25 # mask possibility
vocab_size = 23 # type of tokens

# Determine if GPU is available
if not torch.cuda.is_available():
    raise ValueError("No available GPU")

# Determine the number of available GPUs
GPUs_used = args.GPUs
GPUs_avail = torch.cuda.device_count()
if GPUs_used > GPUs_avail:
    raise ValueError("Available GPU({}) is less than the input value".format(GPUs_avail))

# Load and Screening data
cdr3_df = pd.read_csv(args.input)
cdr3_df["tcr_len"] = [len(i) for i in cdr3_df["cdr3"].tolist()]
cdr3_df = cdr3_df[(cdr3_df["tcr_len"] <= 30) & (cdr3_df["tcr_len"] >= 10)]
all_cdr3s = cdr3_df["cdr3"].to_list()
random.shuffle(all_cdr3s)
train_cdr3s = all_cdr3s[:-200000]
val_cdr3s = all_cdr3s[-200000:]

# a put-back selection of 1,000,000 bars TCR
def get_random_cdr3(cdr3s, num = 1000000):
    cdr3_indexs = random.sample(range(len(train_cdr3s)),num)
    random_cdr3s = []
    for index in cdr3_indexs:
        random_cdr3s.append(cdr3s[index])
    return random_cdr3s

# Map amino acids to serial numbers
def aa_to_index(cdr3):
    new_cdr3 = []
    for aa in cdr3:
        new_cdr3.append(AA[aa])
    return new_cdr3

# creat the Dataloader
def make_data(cdr3s):
    batch = []
    for cdr3 in cdr3s:
        # MASK LM
        input_ids = aa_to_index(cdr3)
        n_pred =  min(max_pred, max(1, math.ceil(len(input_ids) * mask_P))) # 25 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)]
        random.shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            P = random.random()
            if P < 0.8:  # 80%
                input_ids[pos] = 21 # make mask
            elif P > 0.9:  # 10%
                index = random.randint(0, 20 - 1) # random index in vocabulary
                input_ids[pos] = index # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([22] * n_pad)

        all_pos = [i for i in range(len(input_ids))]
        remain_pos = list(set(all_pos) - set(cand_maked_pos))

        # Zero Padding (100% - 25%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            remove_pos = np.random.choice(remain_pos, n_pad, replace=False)
            masked_pos.extend(remove_pos)
            masked_tokens.extend([22] * n_pad)

        batch.append([input_ids, masked_tokens, masked_pos])
    input_ids, masked_tokens, masked_pos  = zip(*batch)
    input_ids, masked_tokens, masked_pos = \
        torch.LongTensor(input_ids), torch.LongTensor(masked_tokens), torch.LongTensor(masked_pos)
    dataset = TensorDataset(input_ids, masked_tokens, masked_pos)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    return loader

# Initialization Model 
model = BERT(n_layers=n_layers, d_model=d_model, \
             n_heads=n_heads, vocab_size=vocab_size, maxlen=maxlen)

# Run the main program
if __name__ =="__main__":

    # Data Parallelism
    model = nn.DataParallel(model, list(range(GPUs_used))) 
    model.cuda()
    net_name = net_name = model.module.__class__.__name__

    # Initializing hyperparameters
    BATCH_SIZE = args.batchsize
    EPOCH = args.max_epoch
    lr = args.lr

    # Learning rate cold start
    SUM_step = 0
    MAX_step = 4000

    # Initializing the loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer to AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Initialized learning rate decay strategy and early stop strategy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min", factor=0.3, patience=2, verbose = True)
    early_stopping = EarlyStopping(patience=4,verbose=True)
    
    # Create the Dataloader for the validation set 
    val_loader = make_data(val_cdr3s)

    for epoch in range(1, EPOCH+1):
        tra_loss = 0.0

        # Create the Dataloader for the training set
        # ps. The TCR is different for each epoch in training set
        cdr3s = get_random_cdr3(train_cdr3s)
        train_loader = make_data(cdr3s)

        # Model switches to training mode
        model.train()
        for tra_step, (input_ids, masked_tokens, masked_pos) in enumerate(train_loader):
            
            # Corrected learning rate according to STEP
            SUM_step += 1
            if SUM_step <= MAX_step:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr*SUM_step/MAX_step

            # Transferring matrices to the GPU        
            input_ids, masked_tokens, masked_pos = input_ids.cuda(), masked_tokens.cuda(), masked_pos.cuda()

            logits_lm, _ = model(input_ids, masked_pos)
            loss_lm = criterion(logits_lm.view(-1, 23), masked_tokens.view(-1))
            loss_lm = (loss_lm.float()).mean()
            tra_loss += loss_lm.detach().cpu() 
            
            # Update parameters via optimizer (which involves gradient cropping)
            optimizer.zero_grad()
            loss_lm.backward()
            clip_grad_norm_(model.parameters(), max_norm=2, norm_type=2)
            optimizer.step()

            # Print Results
            if (tra_step + 1) % 20 == 0:
                print('Train Epoch: {} all_loss = {:.6f} masked loss = {:.6f}'.format(epoch, tra_loss/(tra_step+1), loss_lm))
        
        # Model switches to validation mode
        model.eval()
        val_loss = 0.0

        # No gradient calculation
        with torch.no_grad():
            for val_step,  (input_ids, masked_tokens, masked_pos) in enumerate(val_loader):
                input_ids, masked_tokens, masked_pos = input_ids.cuda(), masked_tokens.cuda(), masked_pos.cuda()

                logits_lm, _ = model(input_ids, masked_pos)
                loss_lm = criterion(logits_lm.view(-1, 23), masked_tokens.view(-1))
                loss_lm = (loss_lm.float()).mean().detach().cpu()
                
                val_loss += loss_lm

                # Print Results
                if (val_step + 1) % 10 == 0:
                    print('Val Epoch: {} all_loss = {:.6f} masked loss = {:.6f}'.format(epoch, val_loss/(val_step+1), loss_lm))
        
        # Determining whether the learning rate needs to be decayed
        scheduler.step(val_loss/(val_step+1))

        # Determining whether to stop training
        early_stopping(val_loss/(val_step+1),model)
        if early_stopping.early_stop:
            print("early stop")
            break

        # save the model
        torch.save(model.state_dict(), args.model_dir)


