import numpy as np 
from sklearn.decomposition import PCA

class Dataset():
    '''
    array - If passing training and test data explicitly, pass it as a n \times d_1 \times d_2...\times d_c \times 2 tensor.
            Else pass the entire dataset as a n \times d_1 \times d_2...\times d_c array,
            where n - #samples, c - #channels, d_i - dimension of i^th channel
    split - train split expressed as a fraction of 1. So 80% train data corresponds to 0.8
    '''
    def __init__(self,array,split):
        if array.shape[-1] == 2:
            self.tr_im_dset = array[0]
            self.ts_im_dset = array[1]
            self.im_dset = np.concatenate((array[0],array[1]))
        else:
            self.im_dset = array
            self.num_samples = array.shape[0]
            tr_rows = np.random.choice(self.num_samples,int(split*self.num_samples),replace=False)
            ts_rows = list(set(range(self.num_samples)) - set(tr_rows))
            self.tr_im_dset = array[tr_rows]
            self.ts_im_dset = array[ts_rows]
        self.num_tr = self.tr_im_dset.shape[0]
        self.num_ts = self.ts_im_dset.shape[0]
        self.tr_im_mean = np.mean(self.tr_im_dset,axis=0)
        self.ts_im_mean = np.mean(self.ts_im_dset,axis=0)
        self.tr_reshape_dset = self.tr_im_dset.reshape(self.num_tr,-1)
        self.ts_reshape_dset = self.ts_im_dset.reshape(self.num_ts,-1)
        self.total_dim = self.ts_reshape_dset.shape[-1]
        self.tr_cent_reshape_dset = (self.tr_im_dset - self.tr_im_mean).reshape(self.num_tr,-1)
        self.ts_cent_reshape_dset = (self.ts_im_dset - self.ts_im_mean).reshape(self.num_ts,-1)
        self.tr_reshape_mean_block = np.tile(self.tr_im_mean.reshape(1,-1),(self.num_tr,1))
        self.ts_reshape_mean_block = np.tile(self.ts_im_mean.reshape(1,-1),(self.num_ts,1))
        pca = PCA(np.minimum(self.num_tr,self.total_dim),svd_solver='full')
        pca.fit(self.tr_cent_reshape_dset)
        self.eigvec = pca.components_.T
        self.eigval = pca.explained_variance_
        self.tr_eig_reshape_dset = np.matmul(self.tr_cent_reshape_dset,self.eigvec)
        self.ts_eig_reshape_dset = np.matmul(self.ts_cent_reshape_dset,self.eigvec)

class Metrics():
    def __init__(self,accuracy=0):
        self.p_distortion = []
        self.e_distortion = []
        self.bits = []
        if accuracy:
            self.accuracy = []
        self.ssim = []
