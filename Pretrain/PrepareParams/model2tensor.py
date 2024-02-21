'''
Extract parameters from the trained model
'''
import sys

sys.path.append('../../Pretrain')

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from datasets import *
from Models.MetaKnowledgeLearner import *
from NetSet import *
from torch.utils.data.sampler import *
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from utils import *

if __name__ == '__main__':

    with open('../config.yaml') as f:
        config = yaml.full_load(f)
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    model = StgnnSet(data_args, task_args, model_args, model='v_GWN')  # v_STGCN5
    ifFirstRegion = True
    # for dataset in ['DC','BM','man']:
    for dataset in ['DC','BM','man']:
        if dataset == 'BM':
            nodenum = 267 #160
        elif dataset == 'DC':
            nodenum = 194 #149
        elif dataset == 'man':
            nodenum = 195 #173

        for nodeindex in tqdm(range(nodenum)):
            
            lengthlist = []
            startlist  = [0]
            shapelist  = []
            model.model = torch.load('../Param/Task4/{}_v_GWN_656_20230922/task4_{}.pt'.format(dataset, nodeindex), map_location=torch.device('cpu'))

            allparams = list(model.model.named_parameters())

            iffirst = True
            for singleparams in allparams:
                astensor = singleparams[1].clone().detach() 
                shapelist.append(astensor.shape)
                tensor1D = astensor.flatten()
                lengthlist.append(tensor1D.shape[0])
                tensor1D = tensor1D.unsqueeze(0)
                if iffirst == True:
                    finaltensor = tensor1D
                    iffirst = False
                else:
                    finaltensor = torch.cat((finaltensor,tensor1D), dim = 1)
                startlist.append(finaltensor.shape[1])
            if ifFirstRegion==True:
                allRegionTensor = finaltensor
                ifFirstRegion = False
            else: 
                allRegionTensor = torch.cat((allRegionTensor, finaltensor), dim=0)
    

    np.save('ModelParams_GWN_91904_656_GUIYIHUA', allRegionTensor.cpu())  
    print(allRegionTensor.shape)
