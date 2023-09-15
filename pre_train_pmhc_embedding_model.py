import torch
import math
import argparse
import torch.nn as nn
import pandas as pd
from embeding import *
from utils import *
import random
from bert_pmhc import BERT
from pytorchtools import EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset

parser = argparse.ArgumentParser(description='train the pmhc embedding model')

# File dir 
parser.add_argument('--input', type=str, default="./data/train_pmhc.csv", help='MHC-pep pairs csv with columns ["allele", "peptide", "label"]')
parser.add_argument('--random_peptide', type=str, default="./data/natural_peptide.csv", help='a csv with random natural peptides with column "peptide"')
parser.add_argument('--model_dir', type=str, default="./output/pmhc_model.pt", help='output dir of model')

# Hyperparameters
parser.add_argument('--n_layers', type=int, default=4, help='number of encoder layers')
parser.add_argument('--d_model', type=int, default=256, help='number of embedding dimention')
parser.add_argument('--neg_X', type=int, default=2, help='negative case multiple')
parser.add_argument('--batchsize', type=int, default=1024, help='mini batchsize')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=100)

# GPUs
parser.add_argument('--GPUs', type=int, default=2, help='num of GPUs used in this task')

args = parser.parse_args()

pos_num = 0
neg_num = 0

# BERT Parameters
maxlen = 54 # max tokens of pmhc sequence
max_pred = 10 # max tokens of prediction
n_layers = args.n_layers  # num of transformer encoder layers
n_heads = 8   # num of muti self-attention heads 
d_model = args.d_model   # embedding dimention
d_ff = d_model*4 # 4*d_model, FeedForward dimension
d_k = d_v = d_model*2  # dimension of K(=Q), V
n_segments = 2 # num of types of segment tokens
neg_X = args.neg_X # negative case multiple
mask_P = 0.25 # mask possibility
vocab_size = 25 # type of tokens

# Determine if GPU is available
if not torch.cuda.is_available():
    raise ValueError("No available GPU")

# Determine the number of available GPUs
GPUs_used = args.GPUs
GPUs_avail = torch.cuda.device_count()
if GPUs_used > GPUs_avail:
    raise ValueError("Available GPU({}) is less than the input value".format(GPUs_avail))

# Load the data
raw_df = pd.read_csv(args.input)
alleles = raw_df["allele"].to_list()
peptides = raw_df["peptide"].to_list()
labels = raw_df["label"].to_list()

# Label transfor
labels = [i if i <= 0.36 else 1.0 for i in labels]

# Load the natural peptide to create negative MHC-pep pairs
random_peptides = pd.read_csv(args.random_peptide)["peptide"].to_list()

# Calculate the number of case
for i in labels:
    if i == 1:
        pos_num += 1
    else:
        neg_num += 1

# If negative case is less than positive case * Negative case multiple \
# create negative case until they are equal
while neg_num < neg_X*pos_num:
    index = random.randint(0, len(raw_df)-1)
    alleles.append(alleles[index])
    random_peptide = random.sample(random_peptides, 1)[0]
    peptides.append(random_peptide)
    labels.append(0)
    neg_num += 1 

# Map amino acids to serial numbers
def aa_to_index(cdr3):
    new_cdr3 = []
    for aa in cdr3:
        new_cdr3.append(AA[aa])
    return new_cdr3

# creat the Dataloader
def make_data(alleles, peptides, labels):
    batch = []
    for allele, peptide, label in zip(alleles, peptides, labels):
        input_ids = aa_to_index(allele) + [23] + aa_to_index(peptide) + [23]    
        segment_ids = [0] * (len(allele) + 1) + [1] * (len(peptide) + 1)
        
        # MASK LM
        n_pred =  min(max_pred, max(1, math.ceil(len(input_ids) * mask_P))) # 25 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                            if token != 23] # candidate masked position
        random.shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            P = random.random()
            if P < 0.8:  # 80%
                input_ids[pos] = 21 # make mask
            elif P > 0.9:  # 10%
                index = random.randint(0, 20) # random index in vocabulary
                input_ids[pos] = index # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([22] * n_pad)
        segment_ids.extend([1] * n_pad)

        all_pos = [i for i in range(len(input_ids))]
        remain_pos = list(set(all_pos) - set(cand_maked_pos))

        # Zero Padding (100% - 25%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            remove_pos = np.random.choice(remain_pos, n_pad, replace=False)
            masked_pos.extend(remove_pos)
            masked_tokens.extend([22] * n_pad)

        batch.append([input_ids, segment_ids, masked_tokens, masked_pos, label])

    input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
        torch.LongTensor(input_ids),  torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), \
            torch.LongTensor(masked_pos), torch.tensor(isNext, dtype=torch.float32)
    loader = DataLoader(TensorDataset(input_ids, segment_ids, masked_tokens, masked_pos, isNext), \
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    return loader

# Split the train data
train_alleles, val_alleles, train_peptides, val_peptides, train_labels, val_labels = \
    train_test_split(alleles, peptides, labels, test_size=0.2, random_state=0)


# Initialization Model 
model = BERT(n_layers=n_layers, d_model=d_model, n_heads=n_heads, \
                 maxlen=maxlen, n_segments=n_segments, vocab_size=vocab_size)


# Run the main program
if __name__ == "__main__":

    # Data Parallelism
    model = nn.DataParallel(model, list(range(GPUs_used))) 
    model.cuda()
    net_name = model.module.__class__.__name__
                
    # Initializing hyperparameters
    BATCH_SIZE = args.batchsize
    EPOCH = args.max_epoch
    lr = args.lr

    # Learning rate cold start
    SUM_step = 0
    MAX_step = 4000
    
    # Initializing the loss function
    mask_loss = nn.CrossEntropyLoss()
    cls_LOSS = nn.MSELoss()

    # Initialize optimizer to AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Initialized learning rate decay strategy and early stop strategy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=4, verbose=True)
    
    # Create the Dataloader for the validation set 
    val_loader = make_data(val_alleles, val_peptides, val_labels)
    
    # Start training
    for epoch in range(1, EPOCH+1):
        tra_loss = 0.0

        # Create the Dataloader for the training set
        # ps. The Dataloader for the training set changes every epoch, 
        # that means each epoch has different masked tokens
        train_loader = make_data(train_alleles, train_peptides, train_labels)

        # Model switches to training mode
        model.train()
        for tra_step, (input_ids, segment_ids, masked_tokens, masked_pos, isNext) in enumerate(train_loader):

            # Corrected learning rate according to STEP
            SUM_step += 1
            if SUM_step <= MAX_step:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr*SUM_step/MAX_step
            
            # Transferring matrices to the GPU
            input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
                input_ids.cuda(), segment_ids.cuda(), masked_tokens.cuda(), masked_pos.cuda(), isNext.cuda()
            
            logits_lm, logits_clsf, _ = model(input_ids, segment_ids, masked_pos)
            
            # for masked token prediction
            loss_lm = mask_loss(weight_loss(isNext, masked_tokens, logits_lm).view(-1, 25), masked_tokens.view(-1))
            loss_lm = (loss_lm.float()).mean() 
            
            # for sentence classification
            loss_clsf = cls_LOSS(logits_clsf.view(-1), isNext) 
            
            # Update parameters via optimizer
            optimizer.zero_grad()
            loss = loss_lm + loss_clsf
            loss.backward()
            optimizer.step()

            pearson = get_pearson_corr(logits_clsf.view(-1), isNext)
            tra_loss += loss.data

            # Print Results
            if (tra_step + 1) % 100 == 0:
                print("-"*20)
                print('Train Epoch:', '%04d' % (epoch), 'loss =', '{:.6f}'.format(tra_loss/(tra_step+1)))
                print('cls loss：{:.6f}， masked loss：{:.6f}，pearson：{:.6f}'.format(loss_clsf, loss_lm, pearson))
        
        # Model switches to validation mode
        model.eval()
        val_loss = 0.0

        # No gradient calculation
        with torch.no_grad():
            for val_step, (input_ids, segment_ids, masked_tokens, masked_pos, isNext) in enumerate(val_loader):
                input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
                    input_ids.cuda(), segment_ids.cuda(), masked_tokens.cuda(), masked_pos.cuda(), isNext.cuda()
                logits_lm, logits_clsf, _ = model(input_ids, segment_ids, masked_pos)
                
                # for masked token prediction
                loss_lm = mask_loss(weight_loss(isNext, masked_tokens, logits_lm).view(-1, 25), masked_tokens.view(-1))
                loss_lm = (loss_lm.float()).mean()

                # for sentence classification
                loss_clsf = cls_LOSS(logits_clsf.view(-1), isNext) 
                loss = loss_lm + loss_clsf

                pearson = get_pearson_corr(logits_clsf.view(-1), isNext)
                val_loss += loss.data

                # Print Results
                if (val_step + 1) % 80 == 0:
                    print('Val Epoch:', '%04d' % (epoch), 'loss =', '{:.6f}'.format(val_loss/(val_step+1)))
                    print('cls loss：{:.6f}， masked loss：{:.6f}，pearson：{:.6f}'.format(loss_clsf, loss_lm, pearson))
        
        # Determining whether the learning rate needs to be decayed
        scheduler.step(val_loss/(val_step+1))

        # Determining whether to stop training
        early_stopping(val_loss/(val_step+1),model)
        if early_stopping.early_stop:
            print("early stop")
            break
    
    # save the model
    torch.save(model.state_dict(), args.model_dir)
    
            
