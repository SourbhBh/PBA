# PBA
Principal Bit Analysis: Autoencoding for Schur-Concave Loss 

This repository contains python code for the ICML 2021 paper "Principal Bit Analysis: Autoencoding for Schur-Concave Loss". The code implements a general purpose, learned, fixed-rate compressor derived from Principal Bit Analysis (PBA).   

## Dependencies
Python >=3.6

Numpy >=1.19.2

glymur >=0.9.2 (for JPEG2000)

PIL >=6.2.1 

PyTorch >=1.0.1

## Usage 

You can view sample code in Example code.ipynb. 

### Dataset and Metrics 
To use this implementation on a custom dataset, pass a dataset where each row is a data point to the 'Dataset' class. 

    from dataset_models import *
    from new_utilities import *
    #custom_dataset_np is dataset as a numpy array
    custom_dataset = Dataset(custom_dataset_np,split=0.8)
    metrics = Metrics(accuracy=1) #accuracy is 1 if downstream classification is required
### Algorithms: PBA and PCA
PBA and PCA can be applied on the dataset. PBA takes as arguments a lambda value that multiplies rate and a hyperparameter a, that controls the width of the clamping interval. PCA takes as arguments the number of components and the hyperparameter a.   
  
    custom_dataset = quant_pba(custom_dataset, lmb, a)
    custom_dataset = quant_pca(custom_dataset, num_components, a)
  
