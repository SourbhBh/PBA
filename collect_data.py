import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import wave
import scipy

def collect_data_mit_faces(data_dir):
    '''
    INPUTS
     - data_dir - full path of data directory
    OUTPUTS
     - im_dataset - Data of shape 3993x128x128
    '''
    directory = os.fsencode(data_dir)
    image_collection = np.zeros((128,128,1))
    count=0
    for file in os.listdir(directory):
        count= count+1
        filename = os.fsdecode(file)
        temp_image = np.fromfile(data_dir+filename, dtype='uint8', sep="")
        ratio = temp_image.shape[0]/16384
        if ratio>1:
            temp_image = np.reshape(temp_image, (128*int(np.sqrt(ratio)), 128*int(np.sqrt(ratio))))
            temp_image = temp_image[:128*int(np.sqrt(ratio)), :128*int(np.sqrt(ratio))].reshape(128, int(np.sqrt(ratio)), 128, int(np.sqrt(ratio))).mean(axis=(1, 3))
        image_collection = np.concatenate((image_collection,np.reshape(temp_image,(128,128,1))),axis=2)
    image_collection = image_collection[:,:,1:] #becomes 128x128x3993
    im_dataset = image_collection.transpose(2,0,1)
    return im_dataset

def collect_data_mnist(folder_path):
    train = np.genfromtxt(folder_path+'mnist_train.csv', delimiter=',')
    test = np.genfromtxt(folder_path+'mnist_test.csv',delimiter=',')
    tr_labels = train[:,0]
    ts_labels = test[:,0]
    ts_data = test[:,1:]
    tr_data = train[:,1:] #csv file first column is label and im_data is 60k x 784
    return tr_data, ts_data, tr_labels, ts_labels

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        return data


def collect_data_cifar10(data_dir):
    train_data = None
    train_labels = []
    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']
            train_labels = data_dic['labels']
        else:
            train_data = np.vstack((train_data, data_dic['data']))
            train_labels += data_dic['labels']

    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.rollaxis(train_data, 1, 4)
    train_labels = np.array(train_labels)

    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)
    test_labels = np.array(test_labels)
    return train_data, train_labels, test_data, test_labels

def collect_data_fsdd(data_dir):
    reshape_dataset = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            rate, audio = scipy.io.wavfile.read(directory+filename)
            if audio.shape[0] >= 6000:
                reshape_dataset.append(audio[:6000])
            else:
                reshape_dataset.append(np.hstack((audio,np.zeros((6000-audio.shape[0],),dtype=audio.dtype))))
    reshape_dataset = np.stack(reshape_dataset,axis=0)
    return reshape_dataset


def split_train_test(train_data,train_labelss,split):
    '''
    INPUTS
     - train_data - Total train data
     - test_data - Total test data for accuracy testing
     - train_labels - Training labels for complete training data
     - test_labels - Test labels
     - split - Fraction of training data to use for accuracy experiment
    OUTPUTS
     - train_data_alg - Training data for compression alg
     - test_tot - Test data for compression alg (concatentation of training data for accuracy experiment and test data for accuracy experiment)
     - train_labels_acc - Training labels for accuracy experiment
    '''
    acc_rows = np.random.choice(train_data.shape[0],int(split*train_data.shape[0]),replace=False)
    alg_rows = list(set(range(train_data.shape[0])) - set(acc_rows))
    train_data_acc = train_data[acc_rows]
    train_labels_acc = train_labels[acc_rows]
    train_data_alg = train_data[alg_rows]
    train_labels_alg = train_labels[alg_rows]
    return train_data_alg, train_data_acc, train_labels_acc
    
