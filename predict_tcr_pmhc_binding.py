import torch
import torch.nn as nn
from embeding import *
from utils import *
import mhcnames
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
from torch.utils.data import DataLoader,TensorDataset

parser = argparse.ArgumentParser(description='predict wheather the tcr and pmhc can bind')

# File dir
parser.add_argument('--input', type=str, default="./data/test_S1_tcr_pmhc.csv", help="input data ,includes \
                    the following three columns:'peptide','allele','cdr3'")
parser.add_argument('--healthy_tcr', type=str, default="./data/small_healthy_tcr.csv", \
                    help='TCR for healthy people csv')
parser.add_argument('--pseudo_sequence_dict', type=str, default="./data/mhcflurry.allele_sequences_homo.csv",\
                     help='allele name to pseudo sequence csv file dir')
parser.add_argument('--output', type=str, default="./output/output.csv",\
                     help='output file')

# model weights 
parser.add_argument('--tcr_pmhc_model', type=str, default="./model/tcr_pmhc_model.pt",\
                     help='TCR-pMHC prediction model dir')
parser.add_argument('--tcr_model', type=str, default="./model/tcr_model.pt",\
                     help='TCR embedding model dir')
parser.add_argument('--pmhc_model', type=str, default="./model/pmhc_model.pt",\
                     help='pMHC embedding model dir')

# Hyperparameters
parser.add_argument('--batchsize', type=int, default=256, help='mini batchsize')
parser.add_argument('--embedding_batchsize', type=int, default=256, \
                    help='mini batchsize of generation embedding')

parser.add_argument('--pmhc_d_model', type=int, default=256, help='dimention of pmhc embedding')
parser.add_argument('--tcr_d_model', type=int, default=256, help='dimention of tcr embedding')

# GPUs 
parser.add_argument('--GPUs', type=int, default=1, help='num of GPUs used in this task, if you have GPU recommend 1, if not, recommend 0')

args = parser.parse_args()

# Determine if GPU is currently available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

# Determine the number of available GPUs
GPUs_used = args.GPUs
GPUs_avail = torch.cuda.device_count()
if GPUs_used == 0 and str(device) != "cpu":
    raise ValueError("Available GPUs are detected, please modify the --GPUs parameter")

if GPUs_used > GPUs_avail:
    raise ValueError("Available GPU({}) is less than the input value".format(GPUs_avail))

input_data_file = args.input
output_file = args.output
healthy_tcr_file = args.healthy_tcr
pseudo_sequence_file = args.pseudo_sequence_dict
tcr_pmhc_model_file = args.tcr_pmhc_model
pmhc_model_file = args.pmhc_model
tcr_model_file = args.tcr_model
pmhc_d_model = args.pmhc_d_model
tcr_d_model = args.tcr_d_model
embedding_BATCH_SIZE = args.embedding_batchsize
BATCH_SIZE = args.batchsize
tcr_maxlen = 30
pmhc_maxlen = 54

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

# Initialize tcr-pmhc prediction model and load weights  
model = tcr_pmhc()
if GPUs_used == 0:
    model = nn.DataParallel(model) 
else:
    model = nn.DataParallel(model, list(range(GPUs_used)))
model.load_state_dict(torch.load(tcr_pmhc_model_file, map_location=device))
model.to(device)
model.eval()

# Initialize two pre-trained models and load weights   
from bert_pmhc import BERT as pmhc_net
from bert_tcr import BERT as tcr_net

pmhc_model = pmhc_net(device=device)
if GPUs_used == 0:
    pmhc_model = nn.DataParallel(pmhc_model)
else:
    pmhc_model = nn.DataParallel(pmhc_model, list(range(GPUs_used)))
pmhc_model.load_state_dict(torch.load(pmhc_model_file, map_location=device))
pmhc_model.to(device)
pmhc_model.eval()

tcr_model = tcr_net(device=device)
if GPUs_used == 0:
    tcr_model = nn.DataParallel(tcr_model)
else:
    tcr_model = nn.DataParallel(tcr_model, list(range(GPUs_used)))
tcr_model.load_state_dict(torch.load(tcr_model_file, map_location=device))
tcr_model.to(device)
tcr_model.eval()

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

# Loading and processing data
allele_dict = pd.read_csv(pseudo_sequence_file)
allele_dict = allele_dict.set_index("allele")

input_df = pd.read_csv(input_data_file)
alleles = input_df["allele"].tolist()
alleles = [mhcnames.normalize_allele_name(allele) for allele in alleles]
peptides = input_df["peptide"].tolist()
cdr3s = input_df["cdr3"].tolist()
# labels = input_df["label"].tolist()
healthy_tcrs = pd.read_csv(healthy_tcr_file, nrows=1000)["cdr3"].tolist()

alleles = [allele_dict.at[allele,"sequence"] for allele in alleles]
pmhc_loader = pmhc_make_data(alleles, peptides)
tcr_loader = tcr_make_data(cdr3s)
healthy_loader = tcr_make_data(healthy_tcrs)

# Generate the embedding matrix for predict the tcr-pmhc
pmhc_output = torch.Tensor().to(device)
for input_ids, segment_ids in pmhc_loader:
    with torch.no_grad():
        input_ids, segment_ids = input_ids.to(device), segment_ids.to(device)
        _, output = pmhc_model(input_ids, segment_ids)
    pmhc_output = torch.cat([pmhc_output,output.reshape(-1, pmhc_maxlen*pmhc_d_model)], dim=0)

tcr_output = torch.Tensor().to(device)
for input_ids in tcr_loader:
    with torch.no_grad():
        input_ids = input_ids[0].to(device)
        output = tcr_model(input_ids)
        tcr_output = torch.cat([tcr_output,output.reshape(-1, tcr_maxlen*tcr_d_model)], dim=0)

healthy_output = torch.Tensor().to(device)
for input_ids in healthy_loader:
    with torch.no_grad():
        input_ids = input_ids[0].to(device)
        output = tcr_model(input_ids)
        healthy_output = torch.cat([healthy_output,output.reshape(-1, tcr_maxlen*tcr_d_model)], dim=0)

TCR_neg_df_1k = healthy_output

def get_rank(test_tcr, test_pmhc, TCR_neg_df_1k, test_label=False):
    preds = []
    ranks = []
    for each_data_index in range(test_tcr.shape[0]):
        tcr_pos=test_tcr[each_data_index].unsqueeze(dim=0)
        pmhc=test_pmhc[each_data_index].unsqueeze(dim=0)
        #used the positive pair with 1k negative tcr to form a 1001 data frame for prediction                                                                      

        TCR_input_df=torch.cat([tcr_pos,TCR_neg_df_1k],dim=0)
        MHC_antigen_input_df= torch.repeat_interleave(pmhc, 1001, dim=0)
        
        prediction = model(torch.cat([TCR_input_df,MHC_antigen_input_df], dim=1))
        preds.append(prediction.tolist()[0])
        rank=1-(sorted(prediction.tolist()).index(prediction.tolist()[0])+1)/1001      
        ranks.append(rank)
    preds = torch.tensor(preds, dtype=torch.float32)
    ranks = [1-i for i in ranks]
    if test_label:
        precision, recall, _thresholds = precision_recall_curve(test_label, ranks)
        PR = auc(recall, precision)
        AUC = roc_auc_score(test_label,ranks)
        return ranks, PR, AUC
    else:
        return ranks


# ranks, PR, AUC = get_rank(tcr_output, pmhc_output, TCR_neg_df_1k, labels)
ranks = get_rank(tcr_output, pmhc_output, TCR_neg_df_1k)
input_df["rank"] = ranks
# print(PR, AUC)
input_df.to_csv(output_file, index=False)
