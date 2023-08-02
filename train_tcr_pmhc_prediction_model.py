import torch
from torch import nn
import pandas as pd
import argparse
from embeding import *
from utils import *
import random
import mhcnames
from pytorchtools import EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(description='train the tcr-pmhc prediction model')

# File dir
parser.add_argument('--input', type=str, default="./data/all_tcr_pmhc.csv", help="input data ,includes \
                    the following three columns:'peptide','allele','cdr3'")
parser.add_argument('--healthy_tcr', type=str, default="./data/small_healthy_tcr.csv", \
                    help='TCR for healthy people csv')
parser.add_argument('--pseudo_sequence_dict', default="./data/mhcflurry.allele_sequences_homo.csv", \
                    type=str, help='allele name to pseudo sequence csv file dir')
parser.add_argument('--model_dir', type=str, default="./output/tcr_pmhc_model.pt", help='where to save the model')

# model weights
parser.add_argument('--tcr_model', type=str, default="./model/tcr_model.pt", help='TCR embedding model dir')
parser.add_argument('--pmhc_model', type=str, default="./model/pmhc_model.pt", help='pMHC embedding model dir')

# Hyperparameters
parser.add_argument('--batchsize', type=int, default=256, help='mini batchsize')
parser.add_argument('--embedding_batchsize', type=int, default=256, \
                    help='mini batchsize of generation embedding')
parser.add_argument('--pmhc_d_model', type=int, default=256, help='dimention of pmhc embedding')
parser.add_argument('--tcr_d_model', type=int, default=256, help='dimention of tcr embedding')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=500)

# GPUs
parser.add_argument('--GPUs', type=int, default=2, help='num of GPUs used in this task')

args = parser.parse_args()

input_data_file = args.input
healthy_tcr_file = args.healthy_tcr
pmhc_d_model = args.pmhc_d_model
tcr_d_model = args.tcr_d_model
pseudo_sequence_file = args.pseudo_sequence_dict
pmhc_model_file = args.pmhc_model
tcr_model_file = args.tcr_model
embedding_BATCH_SIZE = args.embedding_batchsize
model_dir = args.model_dir
pmhc_maxlen = 54
tcr_maxlen = 30

# Determine if GPU is available
if not torch.cuda.is_available():
    raise ValueError("No available GPU")

# Determine the number of available GPUs
GPUs_used = args.GPUs
GPUs_avail = torch.cuda.device_count()
if GPUs_used > GPUs_avail:
    raise ValueError("Available GPU({}) is less than the input value".format(GPUs_avail))

# Loading and processing data
df = pd.read_csv(input_data_file)
df = df[df["train_test"] == "train"]
cdr3s = df["cdr3"].tolist()
alleles = df["allele"].tolist()
alleles = [mhcnames.normalize_allele_name(allele) for allele in alleles]
peptides = df["peptide"].tolist()

# The TCR for generating negative cases in the test set comes from after 10,000 lines
# which is taken to avoid getting duplicate TCRs
healthy_tcrs = pd.read_csv(healthy_tcr_file, nrows=10000)["cdr3"].tolist()

# Map amino acids to serial numbers
def aa_to_index(cdr3):
    new_cdr3 = []
    for aa in cdr3:
        new_cdr3.append(AA[aa])
    return new_cdr3

# creat tcrs' Dataloader    
def tcr_make_data(cdr3s):
    all_inputs = []
    for cdr3 in cdr3s:
        # MASK LM
        input_ids = aa_to_index(cdr3)
        n_pad = tcr_maxlen - len(input_ids)
        input_ids.extend([22] * n_pad)
        all_inputs.append(input_ids)

    dataset = TensorDataset(torch.LongTensor(all_inputs))
    loader = DataLoader(dataset, batch_size=embedding_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    return loader

# creat pmhcs' Dataloader   
def pmhc_make_data(alleles, peptides):
    batch = []
    for allele, peptide in zip(alleles, peptides):
        input_ids = aa_to_index(allele) + [23] + aa_to_index(peptide) + [23]
        segment_ids = [0] * (len(allele) + 1) + [1] * (len(peptide) + 1)

        # Zero Paddings
        n_pad = pmhc_maxlen - len(input_ids)
        input_ids.extend([22] * n_pad)
        segment_ids.extend([1] * n_pad)

        batch.append([input_ids, segment_ids])

    input_ids, segment_ids = zip(*batch)
    input_ids, segment_ids = \
        torch.LongTensor(input_ids),  torch.LongTensor(segment_ids)
    loader = DataLoader(TensorDataset(input_ids, segment_ids), \
        batch_size=embedding_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    return loader

# Mapping genes into pseudo-sequences
allele_dict = pd.read_csv(pseudo_sequence_file)
allele_dict = allele_dict.set_index("allele")
alleles = [allele_dict.at[allele,"sequence"] for allele in alleles]

# Generate dataloader
pmhc_loader = pmhc_make_data(alleles, peptides)
tcr_loader = tcr_make_data(cdr3s)
healthy_loader = tcr_make_data(healthy_tcrs)

# TCR-pMHC prediction model
class tcr_pmhc(nn.Module):
    def __init__(self):
        super(tcr_pmhc, self).__init__()
        self.pmhc_linear = nn.Sequential(
            nn.Linear(pmhc_d_model, 1),
            )
        self.tcr_linear = nn.Sequential(
            nn.Linear(pmhc_d_model, 1),
            )
        
        self._pmhc_linear = nn.Sequential(
            nn.Linear(pmhc_maxlen, 1),
            )
        self._tcr_linear = nn.Sequential(
            nn.Linear(tcr_maxlen, 1),
            )

        self.dense = nn.Sequential(
            nn.Linear((pmhc_maxlen + tcr_maxlen + tcr_d_model + pmhc_d_model) * 1, 200),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(200,100),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(50,1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.reshape(-1, pmhc_maxlen + tcr_maxlen, pmhc_d_model)
        tcr = x[:,:tcr_maxlen,:]
        pmhc = x[:,tcr_maxlen:,:]

        tcr_x = self.tcr_linear(tcr).reshape(-1, tcr_maxlen*1)
        pmhc_x = self.pmhc_linear(pmhc).reshape(-1, pmhc_maxlen*1)

        _tcr_x = self._tcr_linear(tcr.transpose(1,2)).reshape(-1, tcr_d_model*1)
        _pmhc_x = self._pmhc_linear(pmhc.transpose(1,2)).reshape(-1, pmhc_d_model*1)

        out = self.dense(torch.cat([tcr_x, _tcr_x, pmhc_x, _pmhc_x], dim=-1))
        return out.view(-1)

from bert_pmhc import BERT as pmhc_net
from bert_tcr import BERT as tcr_net

# Initialize pmhc pre-trained model and load weights  
pmhc_model = pmhc_net()
pmhc_model = nn.DataParallel(pmhc_model, list(range(GPUs_used)))
pmhc_model.load_state_dict(torch.load(pmhc_model_file))
pmhc_model.cuda()
pmhc_model.eval()

# Generate the embedding matrix for training the tcr-pmhc prediction model
pmhc_output = torch.Tensor()
for input_ids, segment_ids in pmhc_loader:
    with torch.no_grad():
        input_ids, segment_ids = input_ids.cuda(), segment_ids.cuda()
        _, output = pmhc_model(input_ids, segment_ids)
    pmhc_output = torch.cat([pmhc_output,output.reshape(-1, pmhc_maxlen*pmhc_d_model).cpu()], dim=0)

# Clear cache
del pmhc_model, output, _
torch.cuda.empty_cache()

# Initialize tcr pre-trained model and load weights
tcr_model = tcr_net()
tcr_model = nn.DataParallel(tcr_model, list(range(GPUs_used)))
tcr_model.load_state_dict(torch.load(tcr_model_file))
tcr_model.cuda()
tcr_model.eval()

# Generate the embedding matrix for training the tcr-pmhc prediction model
tcr_output = torch.Tensor()
for input_ids in tcr_loader:
    with torch.no_grad():
        input_ids = input_ids[0].cuda()
        output = tcr_model(input_ids)
        tcr_output = torch.cat([tcr_output,output.reshape(-1, tcr_maxlen*tcr_d_model).cpu()], dim=0)

healthy_output = torch.Tensor()
for input_ids in healthy_loader:
    with torch.no_grad():
        input_ids = input_ids[0].cuda()
        output = tcr_model(input_ids)
        healthy_output = torch.cat([healthy_output,output.reshape(-1, tcr_maxlen*tcr_d_model).cpu()], dim=0)

# Delineate the background TCR and the tcr embedding used to generate negative cases for the training set
TCR_neg_df_1k = healthy_output[:1000, :]
healthy_tcrs_matrix = healthy_output[1000:, :]

# Clear cache
del tcr_model, output, healthy_output
torch.cuda.empty_cache()

print("embedding generation is complete")


# Loss function for training the tcr-pmhc prediction model
class TCR_PMHC_loss(nn.Module):
    def __init__(self):
        super(TCR_PMHC_loss, self).__init__()
        self.relu = nn.ReLU()
        self.BCE = nn.BCELoss()
        self.MSE = nn.MSELoss()
    def forward(self, pos_pre, pos_label, neg_pre, neg_label):
        loss = torch.mean(self.relu(1+neg_pre-pos_pre)) + 0.2*torch.mean(neg_pre**2+pos_pre**2)
        return loss

# Generate dataloader for training the tcr-pmhc prediction model
def get_data(tcrs, pmhcs, healthy_tcrs_matrix, mode = "healthy", batch_size = 1024):
    pos = 0
    raw_tensor = torch.cat([tcrs, pmhcs], dim=-1)
    pos_train_tensor = raw_tensor
    neg_train_tensor = torch.zeros_like(pos_train_tensor, dtype = torch.float32)
    for _index, pmhc in enumerate(pmhcs):
        index = 0
        while index < 1:
            if mode == "healthy":
                tcr = healthy_tcrs_matrix[random.randint(0, len(healthy_tcrs_matrix)-1)]
            else:
                tcr = tcrs[random.randint(0, len(tcrs)-1)]

            if tcr == tcrs[_index].all:
                continue
            neg_train_tensor[pos] = torch.cat([tcr, pmhc])
            pos += 1
            index += 1

    data = TensorDataset(pos_train_tensor, neg_train_tensor, torch.full([len(pos_train_tensor)], 1, dtype = torch.float32), \
                         torch.full([len(pos_train_tensor)], 0, dtype=torch.float32))
    loader = DataLoader(data, batch_size=batch_size ,shuffle=True, num_workers=0, pin_memory=True)
    del pos_train_tensor, raw_tensor, neg_train_tensor, data
    return loader


# Run the main program
if __name__ == "__main__":

    # Split the training set and validation set
    train_tcr, val_tcr, train_pmhc, val_pmhc \
            = train_test_split(tcr_output, pmhc_output, test_size=0.1, random_state=0)

    # Initialize tcr-pmhc prediction model 
    model = tcr_pmhc()
    weight_init(model)
    model = nn.DataParallel(model, list(range(GPUs_used))) 
    model.cuda()

    # Initializing the loss function
    LOSS = TCR_PMHC_loss()

    # Initializing hyperparameters
    BATCH_SIZE = args.batchsize
    EPOCH = args.max_epoch
    lr = args.lr

    # Initialization optimizer to AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Initialized learning rate decay strategy and early stop strategy
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min", factor=0.3, \
                                                           patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=6,verbose=True)

    # Create the Dataloader for the validation set 
    val_loader = get_data(val_tcr, val_pmhc, healthy_tcrs_matrix, batch_size=BATCH_SIZE)
    del val_tcr, val_pmhc

    for epoch in range(EPOCH):
        tra_loss = 0.0

        # Create the Dataloader for the training set
        # ps. The negative case is different for each 5 epoch in training set
        if epoch%5 == 0:
            train_loader = get_data(train_tcr, train_pmhc, healthy_tcrs_matrix, batch_size=BATCH_SIZE)

        # Model switches to training mode
        model.train()
        
        for tra_step, (pos, neg, pos_label, neg_label) in enumerate(train_loader):
            pos, neg, pos_label, neg_label = pos.cuda(), neg.cuda(), pos_label.cuda(), neg_label.cuda()

            # Pass in separate positive and negative cases for comparative learning
            pos_pred = model(pos)
            neg_pred = model(neg)
            loss = LOSS(pos_pred, pos_label, neg_pred, neg_label)
            
            # Calculate accuracy
            acc = accuracy_func(torch.cat([pos_pred,neg_pred]),torch.cat([pos_label,neg_label]), 0)

            tra_loss += loss.data

            # Update parameters via optimizer 
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            # Print Results
            if tra_step%200 == 0:
                print(" epoch: {} step: {}  loss: {:.3f} acc: {:.3f}".format(epoch+1, tra_step+1, loss.data, acc))

        # Model switches to validation mode
        model.eval()
        val_loss = 0.0

        # No gradient calculation
        with torch.no_grad():
            for val_step, (pos, neg, pos_label, neg_label) in enumerate(val_loader):
                pos, neg, pos_label, neg_label = pos.cuda(), neg.cuda(), pos_label.cuda(), neg_label.cuda()

                pos_pred = model(pos)
                neg_pred = model(neg)
                loss = LOSS(pos_pred, pos_label, neg_pred, neg_label)          
              
                # Calculate accuracy
                acc = accuracy_func(torch.cat([pos_pred,neg_pred]),torch.cat([pos_label,neg_label]), 0)

                val_loss += loss.data
                
                # Print Results
                if val_step%50 == 0:
                    print(" epoch: {} step: {}  loss{:.3f} acc: {:.3f}".format(epoch+1, \
                        val_step+1, loss.data, acc))
        
        # Determining whether the learning rate needs to be decayed
        scheduler.step(val_loss/(val_step+1))

        # Determining whether to stop training
        early_stopping(val_loss/(val_step+1),model)
        if early_stopping.early_stop:
            break

        # save the model
        torch.save(model.state_dict(), model_dir)