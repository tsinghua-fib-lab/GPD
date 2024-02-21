import sys

import numpy as np
import torch
from numpy import random
from sklearn.datasets import make_moons


def datapreparing(targetDataset, basemodel):
            
    if targetDataset=='None':
        trainid = list(range(0, 656))
        genid = list(range(0, 656))
    elif targetDataset=='DC':
        trainid = list(range(194, 656))
        genid = list(range(0, 194))
    elif targetDataset=='BM':
        trainid = list(range(461, 656)) + list(range(0, 194))
        genid = list(range(194, 461))
    elif targetDataset=='man':
        trainid = list(range(0, 461))
        genid = list(range(461, 656))
    elif targetDataset=='trafficNone':    
        trainid = list(range(0, 1683))
        genid = list(range(0, 1683))
    elif targetDataset == 'metr-la':
        trainid = list(range(207, 1683))
        genid = list(range(0, 207))
    elif targetDataset == 'pems-bay':
        trainid = list(range(0, 207)) + list(range(532, 1683))
        genid = list(range(207, 532))
    elif targetDataset == 'shenzhen':
        trainid = list(range(0, 532)) + list(range(1159, 1683))
        genid = list(range(532, 1159))
    elif targetDataset == 'chengdu_m':
        trainid = list(range(0, 1159))
        genid = list(range(1159, 1683))

    # load full pretrained params
    if basemodel=='v_GWN': 
        rawdata = np.load('../Data/ModelParams_GWN_91904_traffic.npy') 
    else: 
        rawdata = np.load('../Data/ModelParams_STGCN_16960_656.npy')

    genTarget = rawdata[genid]
    training_seq = rawdata[trainid]
    if basemodel=='v_GWN':
        channel = 256  # turn params to 1D sequence, the channel depends on the model dim 
    else:
        channel = 64
    repeatNum = 30

    training_seq = training_seq.reshape(training_seq.shape[0],channel,-1)
    genTarget = genTarget.reshape(genTarget.shape[0],channel,-1)
    training_seq = np.repeat(training_seq, repeatNum, axis=0)

    scale = np.max(np.abs(training_seq))  
    print('larger than 1', np.sum(np.abs(rawdata)>1))

    if scale < 1:
        scale = 1
    scale = 1
    training_seq = training_seq/scale  
    training_seq = training_seq.astype(np.float32)
    genTarget = genTarget/scale 
    genTarget = genTarget.astype(np.float32)

    if targetDataset in ['metr-la', 'pems-bay', 'shenzhen', 'chengdu_m', 'TrafficNone']:
        kgEmb = np.ones((1683, 128))
    else:
        kgEmb = np.load('../Data/Emb/KGEmb.npy')  
    kgEmb = kgEmb.astype(np.float32)

    kgtrainEmb = kgEmb[trainid]
    kggenEmb = kgEmb[genid]
    kgtrainEmb= np.repeat(kgtrainEmb, repeatNum, axis=0)   
    
    if targetDataset in ['metr-la', 'pems-bay', 'shenzhen', 'chengdu_m', 'TrafficNone']:
        timeEmb = np.load('../Data/Emb/trafficTimeEmb.npy')
    else:
        timeEmb = np.load('../Data/Emb/CrowdTimeEmb.npy')
        
    timeEmb = timeEmb.astype(np.float32)
    timetrainEmb = timeEmb[trainid]
    timegenEmb = timeEmb[genid]
    timetrainEmb = np.repeat(timetrainEmb, repeatNum, axis=0)
    
    return training_seq, scale, kgtrainEmb, kggenEmb, timetrainEmb, timegenEmb, genTarget
    
    
