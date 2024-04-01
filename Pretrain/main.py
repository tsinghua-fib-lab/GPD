import argparse
import sys
import time

import setproctitle
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


def getNewIndex(nodeindex, addr):
    A = np.load(addr)
    nodeset = []
    for i in range(A.shape[0]):
        if A[i][nodeindex] != 0  or  A[nodeindex][i] != 0:
            nodeset.append(i)
    for i in range(len(nodeset)):
        if nodeset[i]==nodeindex:
            return i

def train_epoch(train_dataloader):
    train_losses = []
    for step, (data, A_wave) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()
        A_wave = A_wave.to(device=args.device)
        A_wave = A_wave.float()
        data = data.to(device=args.device)
        out, meta_graph = model(data, A_wave)
        loss_predict = loss_criterion(out, data.y)
        loss_reconsturct = loss_criterion(meta_graph, A_wave)
        loss = loss_predict + loss_reconsturct
        loss.backward()
        optimizer.step()
        # print("loss_predict: {}, loss_reconsturct: {}".format(loss_predict.detach().cpu().numpy(), loss_reconsturct.detach().cpu().numpy()))
        train_losses.append(loss.detach().cpu().numpy())
    return sum(train_losses)/len(train_losses)

def test_epoch(test_dataloader):
    with torch.no_grad():
        model.eval()
        for step, (data, A_wave) in enumerate(test_dataloader):
            A_wave = A_wave.to(device=args.device)
            data = data.to(device=args.device)
            out, _ = model(data, A_wave)
            if step == 0:
                outputs = out
                y_label = data.y
            else:
                outputs = torch.cat((outputs, out))
                y_label = torch.cat((y_label, data.y))
        outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
        y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
    return outputs, y_label


parser = argparse.ArgumentParser(description='MAML-based')
parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='metr-la', type=str)
parser.add_argument('--meta_dim', default=32, type=int)
parser.add_argument('--target_days', default=15, type=int)
parser.add_argument('--model', default='v_STGCN', type=str)
parser.add_argument('--loss_lambda', default=1.5, type=float)
parser.add_argument('--memo', default='revise', type=str)
parser.add_argument('--epochs', default=30, type=int)   
parser.add_argument('--taskmode', default='task2', type = str)
parser.add_argument('--nodeindex', default=0, type = int)
parser.add_argument('--iftest', default=True, type = bool)
parser.add_argument("--ifchosenode", action="store_true")
parser.add_argument('--logindex', default='0', type = str)
parser.add_argument('--ifspatial',default=1, type = int)  
parser.add_argument('--ifnewname',default=0, type = int)
parser.add_argument('--aftername',default='', type = str) 
parser.add_argument('--datanum',default=0.7, type = float)  

args = parser.parse_args()

print(time.strftime('%Y-%m-%d %H:%M:%S'), "meta_dim = ", args.meta_dim,"target_days = ", args.target_days)



if __name__ == '__main__':  

    logger, filename = setup_logger(args.taskmode, args.test_dataset, args.logindex, args.model, args.aftername)
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    ifWarp = ""
    if os.path.getsize(filename) != 0:        
        ifWarp = "\n\n"
    logger.info(ifWarp + str(current_time) + ": start training")
    logger.info("target dataset: %s" % args.test_dataset)
    # logger.info(parser.parse_args())
    logger.info("model"+args.model)
    logger.info("taskmode"+args.taskmode)
    logger.info("ifchosenode"+str(args.ifchosenode))
    logger.info("logindex"+str(args.logindex))


    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        logger.info("INFO: GPU")
    else:
        args.device = torch.device('cpu')
        logger.info("INFO: CPU")

    with open(args.config_filename) as f:
        config = yaml.full_load(f)

    # torch.manual_seed(10)

    data_args, task_args, model_args = config['data'], config['task'], config['model']
    
    model_args['meta_dim'] = args.meta_dim
    model_args['loss_lambda'] = args.loss_lambda
    
    logger.info("batchsize: "+str(task_args['batch_size']))
    logger.info("lr: "+str(model_args['meta_lr']))

    loss_criterion = nn.MSELoss()

    source_training_losses, target_training_losses = [], []
    best_result = ''
    min_MAE = 10000000


    if args.taskmode == 'task4':
        '''
        task4: pretrain node-level model
        '''
        node_num = 0
        epochs = 0
        if args.test_dataset == 'bj':
            node_num = 661 # 1010
            epochs = args.epochs
        elif args.test_dataset == 'DC':
            node_num = 194 # 237
            epochs = args.epochs
        elif args.test_dataset == 'BM':
            node_num = 267  #403
            epochs = args.epochs
        elif args.test_dataset == 'man':
            node_num = 195  #280
            epochs = args.epochs
        elif args.test_dataset == 'metr-la' or args.test_dataset == 'pems-bay' or args.test_dataset == 'shenzhen' or args.test_dataset == 'chengdu_m':
            node_num = data_args[args.test_dataset]['node_num']
            epochs = args.epochs
 
        step = 0
        print('{} totalnum: {}'.format(args.test_dataset, node_num))
        for node_index in range(node_num):

            model = StgnnSet(data_args, task_args, model_args, model=args.model).to(device=args.device)
          

            logger.info("train node_index: {}".format(node_index))
            
            train_dataset = traffic_dataset2(data_args, task_args, node_index, "task2", args.ifchosenode, test_data=args.test_dataset, target_days=args.target_days, ifspatial = args.ifspatial, datanum = args.datanum)
            
            train_meanstd = [train_dataset.mean,train_dataset.std]
            
            train_dataloader = DataLoader(train_dataset, batch_size=task_args['batch_size'], shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
            test_dataset = traffic_dataset2(data_args, task_args, node_index, "test", args.ifchosenode, test_data=args.test_dataset, ifspatial = args.ifspatial)
            
            test_meanstd = [test_dataset.mean, test_dataset.std]

            test_dataloader = DataLoader(test_dataset, batch_size=task_args['test_batch_size'], shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
            if args.test_dataset=='BM' or  args.test_dataset=='DC' or  args.test_dataset=='man' or  args.test_dataset=='bj':
                if  args.ifspatial == 1:
                    AAddr = data_args[args.test_dataset]['adjacency_matrix_path']
                else:
                    AAddr = data_args[args.test_dataset]['nonspatial_adjacency_matrix_path']
            else:
                AAddr = data_args[args.test_dataset]['adjacency_matrix_path'] if  args.ifchosenode==True else data_args[args.test_dataset]['iadjacency_matrix_path']
            newindex = getNewIndex(node_index,AAddr)
            # print("newindex: "+ str(newindex))
            outputs, y_label = model.node_taskTrain(args.model, newindex, node_index, 
                                                    train_dataloader, test_dataloader,
                                                    train_meanstd, test_meanstd,
                                                    epochs, logger, args.test_dataset, 
                                                    args.ifnewname, args.aftername)  
            

            
            if step == 0 :
                out = outputs
                truth = y_label
                step = 1
            else:
                out = np.concatenate((out, outputs),axis=2)
                truth = np.concatenate((truth, y_label),axis=2)
        
        logger.info("######################################")
        logger.info("###########  final result  ###########")
        result = metric_func(pred=out, y=truth, times=task_args['pred_num'])
        result_print(result, logger,info_name='Evaluate')


    elif args.taskmode == 'task7':
        '''
        task7: after diffusion sample, finetune
        '''
        node_num = 0
        epochs = 0
        if args.test_dataset == 'bj':
            node_num = 661 # 1010
            epochs = args.epochs
        elif args.test_dataset == 'DC':
            node_num = 194 # 237
            epochs = args.epochs
            params = np.load('DiffusionSample/sampleRes_DC.npy')
            print(params.shape)
        elif args.test_dataset == 'BM':
            node_num = 267  #403
            epochs = args.epochs
            params = np.load('DiffusionSample/sampleRes_BM.npy')
            print('BM!', params.shape)
        elif args.test_dataset == 'man':
            node_num = 195  #280
            epochs = args.epochs
            params = np.load('DiffusionSample/sampleRes_man.npy')
            print(params.shape)
        step = 0
        # node_set = [626]  
        print('totalnum ',node_num)
        params = params.reshape((params.shape[0],-1))
        
        for node_index in tqdm(range(node_num)):
            # node_index = 5
     
            model = StgnnSet(data_args, task_args, model_args, model=args.model).to(device=args.device)

            logger.info("train node_index: {}".format(node_index))
            

            train_dataset = traffic_dataset2(data_args, task_args, node_index, "target", args.ifchosenode, test_data=args.test_dataset, target_days=args.target_days, ifspatial = args.ifspatial, datanum = args.datanum)
            
            train_meanstd = [train_dataset.mean,train_dataset.std]
            
            train_dataloader = DataLoader(train_dataset, batch_size=task_args['batch_size'], shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
            test_dataset = traffic_dataset2(data_args, task_args, node_index, "test", args.ifchosenode, test_data=args.test_dataset, ifspatial = args.ifspatial)
            
            test_meanstd = [test_dataset.mean, test_dataset.std]

            test_dataloader = DataLoader(test_dataset, batch_size=task_args['test_batch_size'], shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
            if args.test_dataset=='BM' or  args.test_dataset=='DC' or  args.test_dataset=='man' or  args.test_dataset=='bj':
                if  args.ifspatial == 1:
                    AAddr = data_args[args.test_dataset]['adjacency_matrix_path']
                else:
                    AAddr = data_args[args.test_dataset]['nonspatial_adjacency_matrix_path']
            else:
                AAddr = data_args[args.test_dataset]['adjacency_matrix_path'] if  args.ifchosenode==True else data_args[args.test_dataset]['iadjacency_matrix_path']
            newindex = getNewIndex(node_index,AAddr)
            # print("newindex: "+ str(newindex))
            param = params[node_index]
            outputs, y_label = model.node_taskTrain2(param, args.model, newindex, 
                                                    node_index, 
                                                    train_dataloader, test_dataloader,
                                                    train_meanstd, test_meanstd,
                                                    epochs, logger, args.test_dataset, 
                                                    args.ifnewname, args.aftername)  

            if step == 0 :
                out = outputs
                truth = y_label
                step = 1
            else:
                out = np.concatenate((out, outputs),axis=2)
                truth = np.concatenate((truth, y_label),axis=2)
        
        logger.info("######################################")
        logger.info("###########  final result  ###########")
        result = metric_func(pred=out, y=truth, times=task_args['pred_num'])
        result_print(result, logger,info_name='Evaluate')
        
        
