
# TABR-BERT

## Introduction
TABR-BERT: an Accurate and Robust BERT-based Transfer Learning Model for TCR-pMHC Interaction Prediction
Contract: hui.yao@freshwindbiotech.com

## Installation
There are two ways to run TABR-BERT

### 1. Docker（recommend）
The Installation of Docker can be seen in https://docs.docker.com/

Pull the image of TABR-BERT from dockerhub:

>docker pull freshwindbioinformatics/tabr-bert:v1

Run the image in bash:

>docker run -it --gpus all freshwindbioinformatics/tabr-bert:v1 bash
#### * Note : The parameter "--gpus" requires docker version higher than 19.03.
#
### 2. Conda and pip

#### Dependencies

-   python == 3.9.12
-   mhcnames == 0.4.8
-   numpy == 1.21.5
-   pandas == 1.2.0
-   scikit_learn == 1.1.3
-   scipy == 1.8.0
-   torch == 1.11.0

#### * Note : If you want to use the GPU, you should install CUDA and cuDNN version compatible with the pytorch version. [Version Searching](https://pytorch.org/)
#
Command:

> conda create -n tabr_bert python==3.9.12  
> conda activate tabr_bert  
> pip install -r requirements.txt  


#### * Note : How to download and install conda? [Documentation](https://docs.conda.io/en/latest/miniconda.html).

<br/>

## Data

You can find the data used to train TCR-BERT, pMHC-BERT and healthy TCR dataset at https://zenodo.org/record/8215354 

## Usage

### Train
#### *Note : If you don't have a GPU, then you can only run the predict file.

#### 1. pretrain TCR embedding model (TCR-BERT)

```
Usage: pre_train_tcr_embedding_model.py [options]
Required:
      --input STRING: The input data to train the TCR embedding model (*.csv) 
                      Required columns: "cdr3"
      --model_dir STRING: where to save the model (*.pt)

Optional:
      --n_layers INT: number of transformer encoder layers (default: 4)
      --d_model INT: number of embedding dimention (default: 256)
      --batchsize INT: mini batchsize (default: 1024)
      --lr Float: learning rate (default: 5e-5)
      --max_epoch INT: Maximum number of train epoch (default: 100)
      --GPUs INT: num of GPUs used in this task(default: 2)
```

#### *Note : If you use docker, then you can train the TCR embedding model directly with the following command:  

>python pre_train_tcr_embedding_model.py

This requires two GPUs with more than 8G of memory, which can reduce the memory requirements by lowering the batchsize, but may affect the stability and effectiveness of training.
#

#### 2. pretrain pMHC embedding model (pMHC-BERT)

```
Usage: pre_train_pmhc_embedding_model.py [options]
Required:
      --input STRING: The input data to train the pMHC embedding model (*.csv) 
                      Required columns: ["allele", "peptide", "label"]
      --random_peptide STRING: natural peptides for generating negative cases (*.csv)
                               Required columns: "peptide"      
      --model_dir STRING: where to save the model (*.pt)

Optional:
      --n_layers INT: number of transformer encoder layers (default: 4)
      --d_model INT: number of embedding dimention (default: 256)
      --neg_X INT: negative case multiple (default: 2)
      --batchsize INT: mini batchsize (default: 1024)
      --lr Float: learning rate (default: 5e-5)
      --max_epoch INT: Maximum number of train epoch (default: 100)
      --GPUs INT: num of GPUs used in this task(default: 2)
```

#### *Note : If you use docker, then you can train the pMHC embedding model directly with the following command:  

>python pre_train_pmhc_embedding_model.py

This requires two GPUs with more than 14G of memory, which can reduce the memory requirements by lowering the batchsize, but may affect the stability and effectiveness of training.
#

#### 3. TCR-pMHC prediction model

```
Usage: train_tcr_pmhc_prediction_model.py [options]
Required:
      --input STRING: The input data to train the TCR-pMHC prediction model (*.csv) 
                      Required columns: ["allele", "peptide", "cdr3"]
      --healthy_tcr STRING: TCRs from healthy people for generating negative cases (*.csv)
                            Required columns: "cdr3" 
      --pseudo_sequence_dict STRING: allele name to pseudo sequence (*.csv)
                                     Required columns: ["allele" "sequence"]    
      --tcr_model STRING: TCR embedding model dir (*.pt)
      --pmhc_model STRING: pMHC embedding model dir (*.pt)                          
      --model_dir STRING: where to save the model (*.pt)

Optional:
      --batchsize INT: mini batchsize (default: 256)
      --embedding_batchsize INT: mini batchsize of generation embedding (default: 256)
      --pmhc_d_model INT: dimention of pmhc embedding (default: 256)
      --tcr_d_model INT: dimention of pmhc embedding (default: 256)
      --lr Float: learning rate (default: 5e-4)
      --max_epoch INT: Maximum number of train epoch (default: 500)
      --GPUs INT: num of GPUs used in this task(default: 2)
```
#### *Note : If you use docker, then you can train the TCR-pMHC prediction model directly with the following command:  

>python train_tcr_pmhc_prediction_model.py

This requires two GPUs with more than 5G of memory, which can reduce the memory requirements by lowering the batchsize, but may affect the stability and effectiveness of training.
#
### Predict
```
Usage: predict_tcr_pmhc_binding.py [options]
Required:
      --input STRING: The data to be predicted (*.csv) 
                      Required columns: ["allele", "peptide", "cdr3"]
      --healthy_tcr STRING: TCRs from healthy people for generating negative cases (*.csv)
                            Required columns: "cdr3" 
      --pseudo_sequence_dict STRING: allele name to pseudo sequence (*.csv)
                                     Required columns: ["allele" "sequence"]   
      --tcr_pmhc_model STRING: TCR-pMHC prediction model dir (*.pt)
      --tcr_model STRING: TCR embedding model dir (*.pt)
      --pmhc_model STRING: pMHC embedding model dir (*.pt)                           
      --output STRING: output file dir (*.csv)

Optional:
      --batchsize INT: mini batchsize (default: 256)
      --embedding_batchsize INT: mini batchsize of generation embedding (default: 256)
      --pmhc_d_model INT: dimention of pmhc embedding (default: 256)
      --tcr_d_model INT: dimention of pmhc embedding (default: 256)
      --GPUs INT: num of GPUs used in this task [if you have GPU recommend 1, if not, recommend 0] (default: 0)
```
#### *Note : If you use docker, then you can predict directly with the following command:  

>python predict_tcr_pmhc_binding.py --input **input_data.csv**

## Citation

You can find the original paper at: https://academic.oup.com/bib/article/25/1/bbad436/7457349
