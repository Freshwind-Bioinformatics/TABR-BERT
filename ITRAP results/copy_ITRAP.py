import pandas as pd
import re
import scipy
import numpy as np
import itertools

df = pd.read_csv("./data/raw_single_cell.csv")
def UMI_filter(df, threshods):
    uca = threshods[0]
    ucb = threshods[1]
    ucm = threshods[2]
    
    # tra trb >= 1
    df = df[(df["umi_count_TRA"] >= uca) & (df["umi_count_TRB"] >= ucb)]
    
    # filter pMHC
    seconds = [i.split("|")[-2] if len(i.split("|"))>1 else i.split("|")[-1] for i in df["umi_count_lst_mhc"].tolist()]
    seconds = [float(i)*1.2 for i in seconds]
    
    df = df[(df["umi_count_mhc"] >= ucm) & (df["umi_count_mhc"] >= seconds)]

    return df

def HLA_match(df):
    df = df[df["HLA_match"]]
    return df


def except_binder(df):
    pep_hlas = set([j for i in df["peptide_HLA_lst"] for j in i.split("|")])
    pep_hla_dict = dict(zip(pep_hlas, [[] for _ in pep_hlas]))
    
    for _, row in df.iterrows():
        pep_hlas = row["peptide_HLA_lst"].split("|")
        umis = [float(i) for i in row["umi_count_lst_mhc"].split("|")]
        for key in pep_hla_dict.keys():
            if key in pep_hlas:
                pep_hla_dict[key].append(umis[pep_hlas.index(key)])
            else:
                pep_hla_dict[key].append(0)
    mean_dict = {key: np.mean(value) for key, value in pep_hla_dict.items()}
    keys = sorted(mean_dict, key=lambda key: mean_dict[key], reverse=True)[:2]
    first = keys[0]
    second = keys[-1]
    stat, p_value = scipy.stats.wilcoxon(pep_hla_dict[first], pep_hla_dict[second])
    if p_value <= 0.05:
        valid_ct = True
        ct_pep = first
    else:
        valid_ct = False
        ct_pep = None
        
    df["_ct_pep"] = [ct_pep for _ in range(len(df))]
    df["_valid_ct"] = [valid_ct for _ in range(len(df))]
    df["_pep_match"] = df["_ct_pep"] == df["peptide_HLA"]
    return df
    
def filter_trainset(df):
    train_df = pd.DataFrame()
    for index, row in df.groupby("ct"):
        if len(row) > 10:
            train_df = pd.concat([train_df, except_binder(row)])
            
    return train_df
    

def grid_search(df, ):
    df = filter_trainset(df)
    umi_count_TRAs = np.arange(1, df.umi_count_TRA.quantile(0.4, interpolation='higher'))
    umi_count_TRBs = np.arange(1, df.umi_count_TRB.quantile(0.4, interpolation='higher'))
    umi_count_mhcs = np.arange(1, df.umi_count_mhc.quantile(0.5, interpolation='higher'))
    
    value_bin = df[~df._ct_pep.isna()].ct.unique() # training set
    best_o = 0
    best_threshold = (0, 0, 0)
    
    for threshold in itertools.product(umi_count_TRAs, umi_count_TRBs, umi_count_mhcs):
        uca = threshold[0]
        ucb = threshold[1]
        ucm = threshold[2]
        
        filter_bool = ((df.umi_count_TRA >= uca) &
                       (df.umi_count_TRB >= ucb) &
                       (df.umi_count_mhc >= ucm))
        flt = df[filter_bool].copy()
        tr_df = df[filter_bool & df.ct.isin(value_bin)].copy()
        acc = tr_df["_pep_match"].mean()
        # frac = flt["_valid_ct"].mean()
        frac = len(flt)/len(df)
        o = (2*acc + frac)/3
        # o = acc
        
        if best_o < o:
            best_o = o
            best_threshold = threshold
        # print(threshold, o)
            
    return best_threshold, best_o
        

# best_threshold, best_o = grid_search(df)
# print(best_threshold)
best_threshold = [1, 1, 5]
ITRAP_df = UMI_filter(df, best_threshold)
ITRAP_df = HLA_match(ITRAP_df)
# ITRAP_df.to_csv("./data/116copy_ITRAP.csv")
ITRAP_df.drop_duplicates(subset=["ct"], inplace=True)
print(len(ITRAP_df))