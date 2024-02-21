# 2023/10/1

import argparse
import os
import sys
import time
from multiprocessing import cpu_count

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from datapreparing import datapreparing
from denoising_diffusion_pytorch import (GaussianDiffusion1D, SimpleDiffusion,
                                         Trainer1D, Unet1D, Unet1D2)
from diffusionutils import *
from numpy import random
from TimeTransformer import (Transformer1, Transformer2, Transformer3,
                             Transformer4, Transformer5)
from TimeTransformer.utils import XEDataset
from torch.utils.data import DataLoader, Dataset
# os.environ['CUDA_VISIBLE_DEVICES']='5' 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.set_num_threads(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100001)
    parser.add_argument("--expIndex", type=int, default=888)
    parser.add_argument("--targetDataset", type=str, default='None')
    parser.add_argument("--diffusionstep", type=int, default=500)
    parser.add_argument("--denoise", type=str, default='Trans3')
    parser.add_argument("--basemodel", type=str, default='v_STGCN5')
    args = parser.parse_args()

    writer = SummaryWriter(log_dir='TensorBoardLogs/exp{}'.format(args.expIndex))

    logger, filename = setup_logger(args.expIndex)
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    ifWarp = ""
    if os.path.getsize(filename) != 0:        
        ifWarp = "\n\n"
    logger.info(ifWarp + str(current_time) + ": begin training")
    
    
    '''
    data preparing
    '''
    training_seq, scale, kgtrainEmb, kggenEmb, timetrainEmb, timegenEmb, genTarget = datapreparing(
        args.targetDataset, args.basemodel
        )  
    print('training_seq.shape', training_seq.shape)
    print('regionEmbedding.shape', kgtrainEmb.shape)  # 128
    print('timeEmbedding.shape', timetrainEmb.shape)  # 128
    print(args)
    XEData = XEDataset(training_seq, kgtrainEmb, timetrainEmb)



    ''' hyper params'''
    denoisingNetworkChoose = args.denoise   # choose denoising network
    experimentIndex = args.expIndex
    deffusionStep = args.diffusionstep
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numEpoch = args.epochs
    batchsize = 64
    sampleTimes = 1
    transformer_dim = args.modeldim
    print('denoisingNetworkChoose:', denoisingNetworkChoose)
    

    '''
    preparing denoising network, many choices
    '''
    
    if denoisingNetworkChoose == 'Trans1':
        '''
            1D sequence simple transformer
        '''
        d_model = transformer_dim  # Lattent dim
        q = 10  # Query size
        v = 10  # Value size
        h = 10  # Number of heads
        N = 3  # Number of encoder and decoder to stack 
        d_kgEmb = kgtrainEmb.shape[1]
        d_timeEmb = timetrainEmb.shape[1]
        attention_size = 5  # Attention window size
        dropout = 0.2  # Dropout rate
        pe = 'original'  # Positional encoding
        chunk_mode = None
        d_input = training_seq.shape[1]  # From dataset
        d_output = d_input  # From dataset
        layernum = training_seq.shape[2]
        ifkg = True
        if args.targetDataset in ['metr-la', 'pems-bay', 'shenzhen', 'chengdu_m', 'TrafficNone']:
            ifkg = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if args.targetDataset in ['metr-la', 'pems-bay', 'shenzhen', 'chengdu_m', 'TrafficNone']:
            spatialloc = [194, 354]
        else:
            spatialloc = [0,4]

        dinoisingModel = Transformer1(d_input, d_model, d_output, d_kgEmb, d_timeEmb, 
                                    q, v, h, N, spatialloc,                                     
                                    attention_size=attention_size, layernum=layernum,                                       
                                    dropout=dropout, chunk_mode=chunk_mode, pe=pe, ifkg = ifkg).to(device)
        
    
    elif denoisingNetworkChoose == 'Trans2':
        '''
            Conditions are added to each layer of transformer
        '''
        
        d_model = transformer_dim  # Lattent dim
        N = 4  # Number of encoder and decoder to stack  
        d_kgEmb = kgtrainEmb.shape[1]
        d_timeEmb = timetrainEmb.shape[1]
        dropout = 0.1  # Dropout rate
        pe = 'original'  # Positional encoding
        d_input = training_seq.shape[1]  # From dataset
        d_output = d_input  # From dataset
        layernum = training_seq.shape[2]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dinoisingModel = Transformer2(d_input, d_model, d_output, d_kgEmb, d_timeEmb, N,                                     
                                    layernum=layernum, dropout=dropout, pe=pe).to(device)  
        
    elif denoisingNetworkChoose == 'Trans3':
        '''
            After the conditions are aggregated, they are added to each layer of transformer
        '''
        
        d_model = transformer_dim  # Lattent dim
        N = 4  # Number of encoder and decoder to stack
        d_kgEmb = kgtrainEmb.shape[1]
        d_timeEmb = timetrainEmb.shape[1]
        dropout = 0.1  # Dropout rate
        pe = 'original'  # Positional encoding
        d_input = training_seq.shape[1]  # From dataset
        d_output = d_input  # From dataset
        layernum = training_seq.shape[2]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dinoisingModel = Transformer3(d_input, d_model, d_output, d_kgEmb, d_timeEmb, N,                                     
                                    layernum=layernum, dropout=dropout, pe=pe).to(device)      

    elif denoisingNetworkChoose == 'Trans4':
        '''
            cross attention
        '''
        
        d_model = transformer_dim  # Lattent dim
        q = 10  # Query size
        v = 10  # Value size
        h = 10  # Number of heads
        N = 4  # Number of encoder and decoder to stack 
        d_kgEmb = kgtrainEmb.shape[1]
        d_timeEmb = timetrainEmb.shape[1]
        attention_size = 5  # Attention window size
        dropout = 0.1  # Dropout rate
        pe = 'original'  # Positional encoding
        chunk_mode = None
        d_input = training_seq.shape[1]  # From dataset
        d_output = d_input  # From dataset
        layernum = training_seq.shape[2]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dinoisingModel = Transformer4(d_input, d_model, d_output, d_kgEmb, d_timeEmb, q, v, h, N,                                     
                                    attention_size=attention_size, layernum=layernum,                                       
                                    dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
      
    elif denoisingNetworkChoose == 'Trans5':
        '''
            Adaptive LayerNorm
        '''
        
        d_model = transformer_dim  # Lattent dim
        q = 10  # Query size
        v = 10  # Value size
        h = 10  # Number of heads
        N = 4  # Number of encoder and decoder to stack  
        d_kgEmb = kgtrainEmb.shape[1]
        d_timeEmb = timetrainEmb.shape[1]
        attention_size = 5  # Attention window size
        dropout = 0.1  # Dropout rate
        pe = 'original'  # Positional encoding  
        chunk_mode = None
        d_input = training_seq.shape[1]  # From dataset
        d_output = d_input  # From dataset
        layernum = training_seq.shape[2]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dinoisingModel = Transformer5(d_input, d_model, d_output, d_kgEmb, d_timeEmb, q, v, h, N,                                     
                                    attention_size=attention_size, layernum=layernum,                                       
                                    dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
      
  

    diffusion = GaussianDiffusion1D( 
        dinoisingModel,  
        seq_length = training_seq.shape[2],
        timesteps = deffusionStep,
        loss_type='l2',      
        objective = 'pred_v',  
        auto_normalize = False,   
        beta_schedule='linear',
    ).to(device)

    
    '''set output sample path'''
    outputpath = './Output/exp{}'.format(args.expIndex)
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    modelsavepath = './ModelSave/exp{}'.format(args.expIndex)
    if not os.path.exists(modelsavepath):
        os.makedirs(modelsavepath)

    trainer = Trainer1D(
        diffusion,
        dataset = XEData,
        train_batch_size = batchsize,
        train_lr = 8e-5,   # 8e-5,
        train_num_steps = numEpoch,             # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        save_and_sample_every = 10,    
        results_folder = modelsavepath,
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,        
        logger = logger,
        kgEmb = kggenEmb,
        timeEmb = timegenEmb,
        genTarget = genTarget,
        targetDataset = args.targetDataset,
        scale = scale,
        tbwriter = writer,
        outputpath = outputpath,
        sampleTimes = sampleTimes,
        basemodel = args.basemodel,
    )
    trainer.train()

    kggenEmb = torch.tensor(kggenEmb).to(device)
    timegenEmb = torch.tensor(timegenEmb).to(device)

    ''' last sample if needed'''

    sampleRes = None
    for _ in range(sampleTimes):
        if args.targetDataset == 'BM':  # to many regions, sample 3 times
            startNum = [0,133,267]
            generateNum = np.diff(startNum)
            result = None
            for id in range(len(generateNum)):
                smallkgEmb = kggenEmb[startNum[id]:startNum[id+1]]
                smalltimeEmb = timegenEmb[startNum[id]:startNum[id+1]]
                sampled_seq = diffusion.sample(
                    torch.tensor(smallkgEmb).to(device), 
                    torch.tensor(smalltimeEmb).to(device), 
                    generateNum[id]
                    ) 
                if result is None:
                    result = sampled_seq.detach().cpu()
                else:
                    result = np.concatenate((result, sampled_seq.detach().cpu()),axis=0)
        elif args.targetDataset in ['DC', 'man', 'metr-la']:
            result = diffusion.sample(
                torch.tensor(kggenEmb).to(device), 
                torch.tensor(timegenEmb).to(device), 
                len(kggenEmb)
                ) 
            result = result.cpu().detach().numpy()
        elif args.targetDataset == 'None':                                    
            startNum = [0,194,461,656]
            generateNum = np.diff(startNum)
            result = None
            for id in range(len(generateNum)):
                smallkgEmb = kggenEmb[startNum[id]:startNum[id+1]]
                smalltimeEmb = timegenEmb[startNum[id]:startNum[id+1]]
                sampled_seq = diffusion.sample(
                    torch.tensor(smallkgEmb).to(device), 
                    torch.tensor(smalltimeEmb).to(device), 
                    generateNum[id]
                    )
                if result is None:
                    result = sampled_seq.detach().cpu()
                else:
                    result = np.concatenate((result, sampled_seq.detach().cpu()),axis=0)
                    
        elif args.targetDataset == 'TrafficNone':                                 
            startNum = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1683]
            generateNum = np.diff(startNum)
            result = None
            for id in range(len(generateNum)):
                smallkgEmb = kggenEmb[startNum[id]:startNum[id+1]]
                smalltimeEmb = timegenEmb[startNum[id]:startNum[id+1]]
                sampled_seq = diffusion.sample(
                    torch.tensor(smallkgEmb).to(device), 
                    torch.tensor(smalltimeEmb).to(device), 
                    generateNum[id]
                    ) 
                if result is None:
                    result = sampled_seq.detach().cpu()
                else:
                    result = np.concatenate((result, sampled_seq.detach().cpu()),axis=0)
                    
        elif args.targetDataset in ['pems-bay', 'shenzhen', 'chengdu_m']:
            length = len(kggenEmb)
            startNum = [0, length//2, length]
            generateNum = np.diff(startNum)
            result = None
            for id in range(len(generateNum)):
                smallkgEmb = kggenEmb[startNum[id]:startNum[id+1]]
                smalltimeEmb = timegenEmb[startNum[id]:startNum[id+1]]
                sampled_seq = diffusion.sample(
                    torch.tensor(smallkgEmb).to(device), 
                    torch.tensor(smalltimeEmb).to(device), 
                    generateNum[id]
                    ) 
                if result is None:
                    result = sampled_seq.detach().cpu()
                else:
                    result = np.concatenate((result, sampled_seq.detach().cpu()),axis=0)
                    
        elif args.targetDataset in ['shenzhen', 'chengdu_m']:
            length = len(kggenEmb)
            startNum = [0, length//3, 2*(length//3), length]
            generateNum = np.diff(startNum)
            result = None
            for id in range(len(generateNum)):
                smallkgEmb = kggenEmb[startNum[id]:startNum[id+1]]
                smalltimeEmb = timegenEmb[startNum[id]:startNum[id+1]]
                sampled_seq = diffusion.sample(
                    torch.tensor(smallkgEmb).to(device), 
                    torch.tensor(smalltimeEmb).to(device), 
                    generateNum[id]
                    ) 
                if result is None:
                    result = sampled_seq.detach().cpu()
                else:
                    result = np.concatenate((result, sampled_seq.detach().cpu()),axis=0)
        
        if sampleRes is None:
            sampleRes = np.expand_dims(result, axis=0)
            # print('sampleRes.shape', sampleRes.shape)
        else:
            sampleRes = np.concatenate((np.expand_dims(result, axis=0), sampleRes),axis=0)
            # print('sampleRes.shape', sampleRes.shape)
            
    sampleRes = np.average(sampleRes, axis = 0)
    np.save('Output/sampleSeq_RealParams_{}'.format(experimentIndex), sampleRes)