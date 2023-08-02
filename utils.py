import numpy as np
import torch 
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score

# calculation pearson correlation coefficient
def get_pearson_corr(y_true, y_pred):
    fsp = y_pred - torch.mean(y_pred)
    fst = y_true - torch.mean(y_true)
    devP = torch.std(y_pred)
    devT = torch.std(y_true)
    return torch.mean(fsp * fst) / (devP * devT)

# calculation spearman correlation coefficient
def get_spearman_corr(y_true, y_pred):
    y_true, y_pred = torch.tensor(rankdata(y_true)),torch.tensor(rankdata(y_pred))
    return get_pearson_corr(y_true, y_pred)

# converting affinity data to ms data
def from_ic50(ic50, max_ic50=50000.0):
    x = 1.0 - (np.log(np.maximum(ic50, 1e-32)) / np.log(max_ic50))
    return np.minimum(
        1.0,
        np.maximum(0.0, x))

# converting ms data to affinity data
def to_ic50(ms, max_ic50=50000.0):
    x = max_ic50 ** (1-ms)
    return x

# calculate accuracy
accuracy_func = lambda y_pred, y_true, threshold: accuracy_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy() > threshold)

# Initialize model weights
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose1d):
        torch.nn.init.normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, torch.nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, torch.nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, torch.nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)

# Determine whether to learn masked token based on label
def weight_loss(label, mask_token, mask_label):
    batch_size, max_pred, token_num = mask_label.shape[0], mask_label.shape[1], mask_label.shape[2]
    max_num = (abs(torch.max(mask_label.detach().cpu()))+1)*100
    one_tensor = torch.full_like(mask_label, 1)
    mul_label = label.repeat(token_num, max_pred, 1).transpose(0,2)
    one_tensor -= mul_label
    one_tensor = one_tensor.reshape(-1, token_num)
    mask_token = mask_token.view(-1)
    for index in range(one_tensor.shape[0]):
        one_tensor[index, int(mask_token[index].data)] += 1
    one_tensor = torch.where(one_tensor > 1, 1, 0)*max_num
    return mask_label + one_tensor.reshape(-1, max_pred, token_num)

import pandas as pd
import pdb
def testset_test(train_df, test_df):
    train_epitope = set(train_df["peptide"].tolist())
    print("训练集共有数据{}条, 共有epitope{}种".format(len(train_df), len(train_epitope)))
    df = pd.DataFrame({"peptide":train_df["peptide"].tolist()+test_df["peptide"].tolist()*2, \
                       "cdr3":train_df["cdr3"].tolist()+test_df["cdr3"].tolist()*2, "allele":train_df["allele"].tolist()+test_df["allele"].tolist()*2})
    df.drop_duplicates(inplace=True, keep=False)
    pdb.set_trace()
    if len(df) == len(train_df):
        print("训练集测试集独立")
    if len(train_epitope) == len(train_epitope-set(test_df["peptide"].tolist())):
        print("epitope zero-shot")