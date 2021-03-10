import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import glymur
import skimage
from PIL import Image
from skimage.measure import compare_ssim as ssim

#Compression Utility Functions
def pca_wrap(centered_tr_data,centered_ts_data,num_components):
    """INPUTS 
       - centered_tr_data : centered training data in a np array of shape num_tr_examples x num_dim
       - centered_ts_data : centered test data in a np array of shape num_ts_examples x num_dim
       - num_components : integer value of numer of components

       OUTPUTS
       - comp_r : np array of shape num_components x num_dim, each row is a prinical component
       - projections : projections of centered_ts_ data onto principal components, np array of shape num_examples x num_components
    """
    pca = PCA(num_components,svd_solver='full')
    pca.fit(centered_tr_data)
    comp_r = pca.components_ #Each row of comp_r is a principal component i.e shape of W - num_components x num_dim
    projections = np.matmul(centered_ts_data,comp_r.T)
    return comp_r,projections

def compute_mse(centered_data,reconst):
    """
    INPUTS 
     - centered_data : centered data in a np array of shape num_examples x num_dim
     - reconst : reconstructed data in a np array of shape num_examples x num_dim
    OUTPUTS 
     - MSE error vec of num_dim  normalized by number of examples
    """
    error = centered_data - reconst    
    return (np.linalg.norm(error,axis=1)**2)/error.shape[0]

def quant(proj,limit_vec):
    '''
    INPUTS
     - proj - np array of num_examples x num_dim  to be quantized
     - limit_vec - width of the interval to quantize each dimension
     OUTPUTS
      - Quantized matrix
    '''
    limit_mat = np.where((-limit_vec<proj) & (proj <=limit_vec),1,0)
    frac = np.sum(limit_mat/proj.shape[0],axis=0)
    return np.where((-limit_vec<proj) & (proj <=limit_vec),np.rint(proj),(np.where(proj<=-limit_vec,np.rint(-limit_vec),np.rint(limit_vec))))

def quant_pca(dataset_pca,num_components,scale=10**11,a=900):
    '''
    INPUTS
     - dataset_pca - Dataset object
     - num_components - integer value of number of components
     - scale - scaling for each component. Default is 10**11 since all significant bits should be sent with full precision
     - a - specifies width of interval to clamp (\sqrt(a)/2 number of std deviations around 0)
    OUTPUTS
     - dataset_pca - Dataset object with reconstructed test datasets in object attributes.  
    '''
    #Project test set data
    pc,proj = pca_wrap(dataset_pca.tr_cent_reshape_dset,dataset_pca.ts_cent_reshape_dset,num_components)
    w_vec = (scale)/np.sqrt(dataset_pca.eigval[:num_components]) #scaling such that all significant components are preserved
    t_vec = (dataset_pca.eigval[:num_components]*w_vec)/(1+dataset_pca.eigval[:num_components]*(w_vec**2))
    dataset_pca.limit_vec = np.sqrt(1+(w_vec**2)*dataset_pca.eigval[:num_components]*a)/2
    proj = np.matmul(proj,np.diag(w_vec))
    quantized_proj = np.matmul(quant(proj,dataset_pca.limit_vec),np.diag(t_vec))
    #Reconstruct image in pixel space
    dataset_pca.ts_cent_reshape_reconst_pca = np.matmul(quantized_proj,pc)
    dataset_pca.ts_reshape_reconst_pca = np.rint(dataset_pca.ts_cent_reshape_reconst + dataset_pca.ts_reshape_mean_block)
    return dataset_pca


def compute_snr_ssim_metrics(dataset,metrics,algorithm,multichannel=False):
    '''
    INPUTS 
     - dataset - Dataset object
     - metrics - Metrics object
     - algorithm - either 'pba' or 'pca'
     - multichannel - For SSIM algorithm. Input True if image is multichannel (eg. rgb) and False otherwise
    OUTPUTS
     - metrics - Updated metrics object with ssim, distortion in eigenspace and pixel space and bits. 
    '''
    if algorithm == 'pca':
        reshape_reconst = dataset.ts_reshape_reconst_pca
        cent_reshape_reconst = dataset.ts_cent_reshape_reconst_pca
        (metrics.bits).append(np.sum(np.log2(2*dataset.limit_vec)))
    elif algorithm == 'pba':
        reshape_reconst = dataset.ts_reshape_reconst_pba
        cent_reshape_reconst = dataset.ts_cent_reshape_reconst_pba
        (metrics.bits).append(dataset.R_min)
    else:
        raise Exception('Field algorithm takes values pba or pca') 
    running_sum = 0
    for i in range(dataset.num_ts):
        running_sum = running_sum + ssim((reshape_reconst[i,:]).reshape((dataset.im_dset).shape[1:]),
                                         (dataset.ts_im_dset[i,:]).astype(np.float64),multichannel=multichannel)
    (metrics.ssim).append(running_sum/dataset.num_ts)
    (metrics.e_distortion).append(np.sum(compute_mse(dataset.ts_cent_reshape_dset,
                                                         cent_reshape_reconst)))
    (metrics.p_distortion).append(np.sum(compute_mse(dataset.ts_reshape_dset,
                                                         reshape_reconst)))
    return metrics

def quant_pba(dataset,lmb,a=900):
    '''
    INPUTS
     - dataset - Dataset object
     - lmb - lambda value (multiplying rate)
     - a - parameter for width of interval (\sqrt(a)/2 number of std deviations around 0)
    OUTPUTS
     - dataset - Updated dataset object. Contains reconstructed test datasets and R_min to update metrics with bits.
    '''
    R = []
    D = []
    d = (dataset.eigval).shape[0]
    #Enumerate potential R and D values 
    for send_comp in range(d):
        c = 1 - (4*lmb*(a-1))/((dataset.eigval)[:send_comp+1])
        c_concave = c[-1]
        if np.sum(c<0) == 0:
            D.append(np.sum(((dataset.eigval)[:send_comp+1]/(2*(a-1)))*(1-np.sqrt(c)))+np.sum((dataset.eigval)[send_comp+1:]/a))
            R.append(np.sum(0.5*np.log2((dataset.eigval)[:send_comp+1]/(4*lmb))) + np.sum(np.log2(1+np.sqrt(c))))
            D_concave = ((dataset.eigval)[send_comp]/(2*(a-1)))*(1+np.sqrt(c_concave))
            if D_concave < (dataset.eigval)[send_comp]/a:
                D.append(np.sum(((dataset.eigval)[:send_comp+1]/(2*(a-1)))*(1-np.sqrt(c))) - ((dataset.eigval)[send_comp]/(2*(a-1)))*(1-np.sqrt(c_concave)) + ((dataset.eigval)[send_comp]/(2*(a-1)))*(1+np.sqrt(c_concave))+np.sum((dataset.eigval)[send_comp+1:]/a))
                R.append(np.sum(0.5*np.log2((dataset.eigval)[:send_comp+1]/(4*lmb))) + np.sum(np.log2(1+np.sqrt(c))) - np.log2(1+np.sqrt(c_concave)) + np.log2(1-np.sqrt(c_concave)))
            else:
                D.append(np.inf)
                R.append(np.inf)
        else:
            D.append(np.inf)
            R.append(np.inf)
    #Identify R and D that minimizes R + \lambda D
    if R != 0 and D != 0:
        min_ind = np.argmin(np.array(D) + lmb*np.array(R))
        R_min=R[min_ind]
        D_min=D[min_ind]
        send_comp = int(min_ind/2) + 1 #actual number of components to send
        #print(send_comp)
        c = 1 - (4*lmb*(a-1))/((dataset.eigval)[:send_comp])
        c_concave = c[-1]
        if min_ind%2 == 0: #convex
            dist_vec = np.concatenate((((dataset.eigval)[:send_comp]/(2*(a-1)))*(1-np.sqrt(c)),(dataset.eigval)[send_comp:]/a),axis=0)
            rate_vec = 0.5*np.log2((dataset.eigval)[:send_comp]/(4*lmb))+np.log2(1+np.sqrt(c))
            w_vec = np.where(dist_vec>0,np.sqrt(1/dist_vec - a/(dataset.eigval)), np.sqrt((np.finfo(np.float64).max)**(1/4) - a/(dataset.eigval)))
            w_vec[(dist_vec==(dataset.eigval)/a)] = 0
            t_vec = ((dataset.eigval)*w_vec)/(1+(dataset.eigval)*(w_vec**2))
            t_vec[(dist_vec==(dataset.eigval)/a)]=0
        else: #concave
            dist_vec = np.concatenate((((dataset.eigval)[:send_comp-1]/(2*(a-1)))*(1-np.sqrt(c)),((dataset.eigval)[send_comp-1]/(2*(a-1)))*(1+np.sqrt(c_concave)),(dataset.eigval)[send_comp:]/a),axis=0)
            rate_vec = np.concatenate((0.5*np.log2((dataset.eigval)[:send_comp-1]/(4*lmb))+np.log2(1+np.sqrt(c)),0.5*np.log2((dataset.eigval)[send_comp])+np.log2(1-np.sqrt(c))))
            w_vec = np.where(dist_vec>0,np.sqrt(1/dist_vec - a/(dataset.eigval)), np.sqrt((np.finfo(np.float64).max)**(1/4) - a/(dataset.eigval)))
            w_vec[(dist_vec==(dataset.eigval)/a)] = 0
            t_vec = ((dataset.eigval)*w_vec)/(1+(dataset.eigval)*(w_vec**2))
            t_vec[(dist_vec==(dataset.eigval)/a)]=0
        eig_proj_pba = np.matmul(dataset.ts_eig_reshape_dset,np.diag(w_vec))
        limit_vec = np.sqrt(1+(w_vec**2)*(dataset.eigval)*a)/2
        eig_quantized_proj_pba = quant(eig_proj_pba,limit_vec)
        eig_reconst_dataset_pba = np.matmul(eig_quantized_proj_pba,np.diag(t_vec))
        dataset.ts_cent_reshape_reconst_pba = np.matmul(eig_reconst_dataset_pba,(dataset.eigvec).T)
        dataset.ts_reshape_reconst_pba = np.rint(dataset.ts_cent_reshape_reconst_pba + dataset.ts_reshape_mean_block)
        dataset.R_min = R_min
        return dataset

def quant_jpegfamily(dataset,metrics,algorithm,qual,multichannel=False):
    '''
    INPUTS
     - dataset - Dataset object
     - metrics - Metrics object
     - algorithm - Takes either 'jpeg' or 'jpeg2k' as values
     - qual - For jpeg this is the quality of the reconstruction, 0 - for lowest quality and 100 for highest quality. 
              For jpeg2k this is the compression ratio which; higher the value more smaller the compressed file.
    OUTPUTS
     - dataset - No updates done to the dataset object in this function as we don't store the reconstructions
     - metrics - Updated metrics object with mse, ssim and bits 
              
    '''

    sum_size = 0
    sum_mse = 0
    sum_ssim = 0
    dataset.ts_reconst_jpeg = np.zeros(dataset.ts_im_dset.shape)
    for i in range(dataset.num_ts):
        curr_im_array = (dataset.ts_im_dset[i,:]).astype(uint8)
        if algorithm == 'jpeg2k':
            jp2 = glymur.Jp2k('trial.jp2',data=curr_im_array,cratios=[qual])
            size_req = os.stat('trial.jp2').st_size*8
        elif algorithm =='jpeg':
            if multichannel==False:
                curr_im = Image.fromarray(curr_im_array,mode='L')
            else:
                curr_im = Image.fromarray(curr_im_array,mode='RGB')
            curr_im.save('trial.jpg',format='jpeg',quality=qual)
        else:
            raise Exception('Not a valid input for algorithm')
        sum_size = sum_size + size_req
        if algorithm == 'jpeg2k':
            jp2_read = glymur.Jp2k('trial.jp2')
            curr_jpg_im_array = jp2_read[:]
        elif algorithm == 'jpeg':
            curr_jpg_im_array = skimage.io.imread('trial.jpg')
        #sum_ssim = sum_ssim + ssim(curr_jpg_im_array,curr_im_array)
        sum_mse = sum_mse + np.linalg.norm(curr_jpg_im_array.astype(np.float64) - curr_im_array.astype(np.float64))**2
        sum_ssim = sum_ssim + ssim(curr_jpg_im_array.astype(np.float64),curr_im_array.astype(np.float64),multichannel=multichannel)
        dataset.ts_reconst_jpeg[i] = curr_jpg_im_array
        if algorithm == 'jpeg2k':
            os.remove('trial.jp2')
        elif algorithm == 'jpeg':
            curr_im.close()
            os.remove('trial.jpg')
    return dataset,metrics        
    

def cifar_compute_accuracy(dataset,train_labels,test_labels,metrics,algorithm,epochs):
    if algorithm == 'pca':
        train_data_acc = (dataset.ts_reshape_reconst_pca.reshape(dataset.ts_im_dset))[:train_labels.shape[0]+1]
        test_data_acc = (dataset.ts_reshape_reconst_pca.reshape(dataset.ts_im_dset))[train_labels.shape[0]+1:]
    elif algorithm == 'pba':
        train_data_acc = (dataset.ts_reshape_reconst_pba.reshape(dataset.ts_im_dset))[:train_labels.shape[0]+1]
        test_data_acc = (dataset.ts_reshape_reconst_pba.reshape(dataset.ts_im_dset))[train_labels.shape[0]+1:]
    elif algorithm == 'jpeg' or 'jpeg2k':
        train_data_acc = (dataset.ts_reconst_jpeg)[:train_labels.shape[0]+1]
        test_data_acc = (dataset.ts_reconst_jpeg)[train_labels.shape[0]+1:]
    (metrics.accuracy).append(cifar_get_accuracy(train_data_acc,train_labels,test_data_acc,test_labels,epochs))
    return metrics 

def mnist_compute_accuracy(dataset,train_labels,test_labels,metrics,algorithm,epochs):
    if algorithm == 'pca':
        train_data_acc = (dataset.ts_reshape_reconst_pca.reshape(dataset.ts_im_dset))[:train_labels.shape[0]+1]
        test_data_acc = (dataset.ts_reshape_reconst_pca.reshape(dataset.ts_im_dset))[train_labels.shape[0]+1:]
    elif algorithm == 'pba':
        train_data_acc = (dataset.ts_reshape_reconst_pba.reshape(dataset.ts_im_dset))[:train_labels.shape[0]+1]
        test_data_acc = (dataset.ts_reshape_reconst_pba.reshape(dataset.ts_im_dset))[train_labels.shape[0]+1:]
    elif algorithm == 'jpeg' or 'jpeg2k':
        train_data_acc = (dataset.ts_reconst_jpeg)[:train_labels.shape[0]+1]
        test_data_acc = (dataset.ts_reconst_jpeg)[train_labels.shape[0]+1:]
    (metrics.accuracy).append(get_accuracy(train_data_acc,train_labels,test_data_acc,test_labels,epochs))
    return metrics 

