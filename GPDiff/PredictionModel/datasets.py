import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from PredictionModel.utils import *
import random
import copy
from torch.utils.data.sampler import *

def loadAM(addr, nodeindex):
    A = np.load(addr)
    nodeset = []
    for i in range(A.shape[0]):
        if A[i][nodeindex] != 0  or A[nodeindex][i] != 0:
            nodeset.append(i)

    n = len(nodeset)
    sm = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            sm[i][j] = A[nodeset[i]][nodeset[j]]

    return sm, nodeset, A

class BBDefinedError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) 
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo

class traffic_dataset(Dataset):
    def __init__(self, data_args, task_args, stage='source', ifchosenode=False, test_data='metr-la', add_target=True, target_days=3):
        super(traffic_dataset, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.stage = stage
        self.add_target = add_target
        self.test_data = test_data
        self.target_days = target_days
        self.load_data(stage, ifchosenode, test_data)
        if self.add_target:
            self.data_list = np.append(self.data_list, self.test_data)
        print("[INFO] Dataset init finished!")

    def load_data(self, stage, ifchosenode, test_data):  
        self.A_list, self.edge_index_list = {}, {}
        self.edge_attr_list, self.node_feature_list = {}, {}
        self.x_list, self.y_list = {}, {}
        self.means_list, self.stds_list = {}, {}

        data_keys = np.array(self.data_args['data_keys'])
        if stage == 'source':
            self.data_list = np.delete(data_keys, np.where(data_keys == test_data))
            # self.data_list = np.array(['metr-la', 'chengdu_m'])
        elif stage == 'task1':
            self.data_list = data_keys  
        elif stage == 'target' or stage == 'target_maml' or stage == 'task2':
            self.data_list = np.array([test_data])
        elif stage == 'test':
            self.data_list = np.array([test_data])
        elif stage == 'dann':
            self.data_list = np.array([test_data])
        else:
            raise BBDefinedError('Error: Unsupported Stage')
        print("[INFO] {} dataset: {}".format(stage, self.data_list))

        for dataset_name in self.data_list:
            AAddr = self.data_args[dataset_name]['adjacency_matrix_path'] 
            A = np.load(AAddr)
            edge_index, edge_attr, node_feature = self.get_attr_func(AAddr)


            self.A_list[dataset_name] = torch.from_numpy(get_normalized_adj(A))
            self.edge_index_list[dataset_name] = edge_index
            self.edge_attr_list[dataset_name] = edge_attr
            self.node_feature_list[dataset_name] = node_feature

            XAddr = self.data_args[dataset_name]['dataset_path'] 
            X = np.load(XAddr)
            X = X.transpose((1, 2, 0))
            X = X.astype(np.float32)
            means = np.mean(X, axis=(0, 2))
            X = X - means.reshape(1, -1, 1)
            stds = np.std(X, axis=(0, 2))
            X = X / stds.reshape(1, -1, 1)

            if stage == 'source' or stage == 'dann':
                X = X
            elif stage == 'task1':
                X = X[:, :, int(X.shape[2]*0.3):]  
            elif stage =='task2':
                X = X[:, :, int(X.shape[2]*0.95):]  
            elif stage == 'target' or stage == 'target_maml':
                X = X[:, :, :288 * self.target_days]
            elif stage == 'test':
                X = X[:, :, :int(X.shape[2]*0.05)]
            else:
                raise BBDefinedError('Error: Unsupported Stage')
            
            x_inputs, y_outputs = generate_dataset(X, self.task_args['his_num'], self.task_args['pred_num'], means, stds)
            
            self.x_list[dataset_name] = x_inputs
            self.y_list[dataset_name] = y_outputs
        
        if stage == 'source' and self.add_target:
            A = np.load(self.data_args[test_data]['adjacency_matrix_path'])
            edge_index, edge_attr, node_feature = self.get_attr_func(
            self.data_args[test_data]['adjacency_matrix_path']
            )

            self.A_list[test_data] = torch.from_numpy(get_normalized_adj(A))
            self.edge_index_list[test_data] = edge_index
            self.edge_attr_list[test_data] = edge_attr
            self.node_feature_list[test_data] = node_feature

            X = np.load(self.data_args[test_data]['dataset_path'])
            X = X.transpose((1, 2, 0))
            X = X.astype(np.float32)
            means = np.mean(X, axis=(0, 2))
            X = X - means.reshape(1, -1, 1)
            stds = np.std(X, axis=(0, 2))
            X = X / stds.reshape(1, -1, 1)

            X = X[:, :, :288 * self.target_days]

            x_inputs, y_outputs = generate_dataset(X, self.task_args['his_num'], self.task_args['pred_num'], means, stds)
            
            self.x_list[test_data] = x_inputs
            self.y_list[test_data] = y_outputs

    def get_attr_func(self, matrix_path, edge_feature_matrix_path=None, node_feature_path=None):
        a, b = [], []
        edge_attr = []
        node_feature = None
        matrix = np.load(matrix_path)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if(matrix[i][j] > 0):
                    a.append(i)
                    b.append(j)
        edge = [a,b]
        edge_index = torch.tensor(edge, dtype=torch.long)

        return edge_index, edge_attr, node_feature
    
    def get_edge_feature(self, edge_index, x_data):
        pass

    def __getitem__(self, index):
        """
        : data.node_num record the node number of each batch
        : data.x shape is [batch_size, node_num, his_num, message_dim]
        : data.y shape is [batch_size, node_num, pred_num]
        : data.edge_index constructed for torch_geometric
        : data.edge_attr  constructed for torch_geometric
        : data.node_feature shape is [batch_size, node_num, node_dim]
        """

        if self.stage == 'source':
            select_dataset = random.choice(self.data_list) 
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
        
        elif self.stage == 'target_maml':
            select_dataset = self.data_list[0]
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]

        else:
            select_dataset = self.data_list[0]
            x_data = self.x_list[select_dataset][index: index+1]
            y_data = self.y_list[select_dataset][index: index+1]

        node_num = self.A_list[select_dataset].shape[0]
        data_i = Data(node_num=node_num, x=x_data, y=y_data)
        data_i.edge_index = self.edge_index_list[select_dataset]
        data_i.data_name = select_dataset
        A_wave = self.A_list[select_dataset]
        return data_i, A_wave
    
    def get_maml_task_batch(self, task_num):  
        spt_task_data, qry_task_data = [], []
        spt_task_A_wave, qry_task_A_wave = [], []

        select_dataset = random.choice(self.data_list)
        batch_size = self.task_args['batch_size']

        for i in range(task_num * 2):
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
            node_num = self.A_list[select_dataset].shape[0]
            data_i = Data(node_num=node_num, x=x_data, y=y_data)
            data_i.edge_index = self.edge_index_list[select_dataset]
            data_i.data_name = select_dataset
            A_wave = self.A_list[select_dataset].float()

            if i % 2 == 0:
                spt_task_data.append(data_i.cuda())
                spt_task_A_wave.append(A_wave.cuda())
            else:
                qry_task_data.append(data_i.cuda())
                qry_task_A_wave.append(A_wave.cuda())

        return spt_task_data, spt_task_A_wave, qry_task_data, qry_task_A_wave


    
    def __len__(self):
        if self.stage == 'source':
            print("[random permutation] length is decided by training epochs")
            return 100000000
        else:
            data_length = self.x_list[self.data_list[0]].shape[0]
            return data_length

class traffic_dataset2(Dataset):
    def __init__(self, data_args, task_args, nodeindex, stage='source', ifchosenode=False, test_data='metr-la', add_target=True, target_days=3, ifspatial=1):
        super(traffic_dataset2, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.stage = stage
        self.add_target = add_target
        self.test_data = test_data
        self.target_days = target_days
        self.nodeindex = nodeindex
        self.mean = 0
        self.std = 0
        self.load_data(stage, ifchosenode, test_data, nodeindex, ifspatial)
        if self.add_target:
            self.data_list = np.append(self.data_list, self.test_data)
        # print("[INFO] Dataset init finished!")

    def load_data(self, stage, ifchosenode, test_data, nodeindex, ifspatial):   
        self.A_list, self.edge_index_list = {}, {}
        self.edge_attr_list, self.node_feature_list = {}, {}
        self.x_list, self.y_list = {}, {}
        self.means_list, self.stds_list = {}, {}

        data_keys = np.array(self.data_args['data_keys'])
        if stage == 'source':
            self.data_list = np.delete(data_keys, np.where(data_keys == test_data))
            # self.data_list = np.array(['metr-la', 'chengdu_m'])
        elif stage == 'task1':
            self.data_list = data_keys  
        elif stage == 'target' or stage == 'target_maml' or stage == 'task2': 
            self.data_list = np.array([test_data])
        elif stage == 'test':
            self.data_list = np.array([test_data])
        elif stage == 'dann':
            self.data_list = np.array([test_data])
        else:
            raise BBDefinedError('Error: Unsupported Stage')

        for dataset_name in self.data_list:
            if dataset_name=='BM' or dataset_name=='DC' or dataset_name=='man' or dataset_name=='bj':
                if ifspatial == 1:
                    AAddr = self.data_args[dataset_name]['adjacency_matrix_path']
                else:
                    AAddr = self.data_args[dataset_name]['nonspatial_adjacency_matrix_path']
            else:
                AAddr = self.data_args[dataset_name]['adjacency_matrix_path'] 
            
            A, nodeset, initA = loadAM(AAddr, nodeindex)
            edge_index, edge_attr, node_feature = self.get_attr_func(A)

            self.A_list[dataset_name] = torch.from_numpy(get_normalized_adj(A))
            # initA2 = torch.from_numpy(get_normalized_adj(initA))
            self.edge_index_list[dataset_name] = edge_index
            self.edge_attr_list[dataset_name] = edge_attr
            self.node_feature_list[dataset_name] = node_feature

            if dataset_name=='BM' or dataset_name=='DC' or dataset_name=='man' or dataset_name=='bj':
                XAddr = self.data_args[dataset_name]['dataset_path']
            else:
                XAddr = self.data_args[dataset_name]['dataset_path'] 

            X = np.load(XAddr)

          
            X = X[:,nodeset,:]
            X = X.transpose((1, 2, 0))
            X = X.astype(np.float32)
            means = np.mean(X, axis=(0, 2))  
            X = X - means.reshape(1, -1, 1)
            stds = np.std(X, axis=(0, 2))
            X = X / stds.reshape(1, -1, 1)

            self.mean = means[0]
            self.std = stds[0]  

            if stage == 'source' or stage == 'dann':
                X = X
            elif stage == 'task1':
                X = X[:, :, int(X.shape[2]*0.3):] 
            elif stage =='task2':
                X = X[:, :, int(X.shape[2]*0.3):]  
            elif stage == 'target' or stage == 'target_maml':
                X = X[:, :, :288 * self.target_days]
            elif stage == 'test': 
                X = X[:, :, :int(X.shape[2]*0.3)] 
            else:
                raise BBDefinedError('Error: Unsupported Stage')
           
            
            x_inputs, y_outputs = generate_dataset(X, self.task_args['his_num'], self.task_args['pred_num'], means, stds)
            
            self.x_list[dataset_name] = x_inputs
            self.y_list[dataset_name] = y_outputs

    def get_attr_func(self, matrix, edge_feature_matrix_path=None, node_feature_path=None):
        a, b = [], []
        edge_attr = []
        node_feature = None
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if(matrix[i][j] > 0):
                    a.append(i)
                    b.append(j)
        edge = [a,b]
        edge_index = torch.tensor(edge, dtype=torch.long)

        return edge_index, edge_attr, node_feature
    
    def get_edge_feature(self, edge_index, x_data):
        pass

    def __getitem__(self, index):
        """
        : data.node_num record the node number of each batch
        : data.x shape is [batch_size, node_num, his_num, message_dim]
        : data.y shape is [batch_size, node_num, pred_num]
        : data.edge_index constructed for torch_geometric
        : data.edge_attr  constructed for torch_geometric
        : data.node_feature shape is [batch_size, node_num, node_dim]
        """

        if self.stage == 'source':
            select_dataset = random.choice(self.data_list) 
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
        
        elif self.stage == 'target_maml':
            select_dataset = self.data_list[0]
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]

        else:
            select_dataset = self.data_list[0]
            x_data = self.x_list[select_dataset][index: index+1]
            y_data = self.y_list[select_dataset][index: index+1]

        node_num = self.A_list[select_dataset].shape[0]
        data_i = Data(node_num=node_num, x=x_data, y=y_data)
        data_i.edge_index = self.edge_index_list[select_dataset]
        data_i.data_name = select_dataset
        A_wave = self.A_list[select_dataset]
        return data_i, A_wave
    
    def get_maml_task_batch(self, task_num):  
        spt_task_data, qry_task_data = [], []
        spt_task_A_wave, qry_task_A_wave = [], []

        select_dataset = random.choice(self.data_list)
        batch_size = self.task_args['batch_size']

        for i in range(task_num * 2):
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
            node_num = self.A_list[select_dataset].shape[0]
            data_i = Data(node_num=node_num, x=x_data, y=y_data)
            data_i.edge_index = self.edge_index_list[select_dataset]
            data_i.data_name = select_dataset
            A_wave = self.A_list[select_dataset].float()

            if i % 2 == 0:
                spt_task_data.append(data_i.cuda())
                spt_task_A_wave.append(A_wave.cuda())
            else:
                qry_task_data.append(data_i.cuda())
                qry_task_A_wave.append(A_wave.cuda())

        return spt_task_data, spt_task_A_wave, qry_task_data, qry_task_A_wave

    
    def __len__(self):
        if self.stage == 'source':
            print("[random permutation] length is decided by training epochs")
            return 100000000
        else:
            data_length = self.x_list[self.data_list[0]].shape[0]
            return data_length
        
    def len(self):
        pass


    def get(self):
        pass


class node_dataset(Dataset):
    def __init__(self, node_index, data_args, task_args, stage='source', test_data='metr-la', add_target=True, target_days=3):
        super(node_dataset, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.stage = stage
        self.add_target = add_target
        self.test_data = test_data
        self.target_days = target_days
        self.load_data(node_index, stage, test_data)
        if self.add_target:
            self.data_list = np.append(self.data_list, self.test_data)
        print("[INFO] Dataset init finished!")

    def load_data(self, node_index, stage, test_data):   
        self.A_list, self.edge_index_list = {}, {}
        self.edge_attr_list, self.node_feature_list = {}, {}
        self.x_list, self.y_list = {}, {}
        self.means_list, self.stds_list = {}, {}

        data_keys = np.array(self.data_args['data_keys'])
        if stage == 'node':
            self.data_list = np.array([test_data])
        elif stage == 'test':
            self.data_list = np.array([test_data])
        else:
            raise BBDefinedError('Error: Unsupported Stage')
        print("[INFO] {} dataset: {}".format(stage, self.data_list))

        for dataset_name in self.data_list:
            A = np.load(self.data_args[dataset_name]['adjacency_matrix_path'])
            edge_index, edge_attr, node_feature = self.get_attr_func(
            self.data_args[dataset_name]['adjacency_matrix_path']
            )

            self.A_list[dataset_name] = torch.from_numpy(get_normalized_adj(A))
            self.edge_index_list[dataset_name] = edge_index
            self.edge_attr_list[dataset_name] = edge_attr
            self.node_feature_list[dataset_name] = node_feature

            X = np.load(self.data_args[dataset_name]['dataset_path'])
            X = X.transpose((1, 2, 0))
            X = X.astype(np.float32)
            means = np.mean(X, axis=(0, 2))
            X = X - means.reshape(1, -1, 1)
            stds = np.std(X, axis=(0, 2))
            X = X / stds.reshape(1, -1, 1)


            X = X[node_index]
            X = X[None]

            if stage =='node':
                X = X[:, :, int(X.shape[2]*0.3):]  
            elif stage == 'test':
                X = X[:, :, :int(X.shape[2]*0.3)]  
            else:
                raise BBDefinedError('Error: Unsupported Stage')
            
            x_inputs, y_outputs = generate_dataset(X, self.task_args['his_num'], self.task_args['pred_num'], means, stds)
            
            self.x_list[dataset_name] = x_inputs
            self.y_list[dataset_name] = y_outputs

    def get_attr_func(self, matrix_path, edge_feature_matrix_path=None, node_feature_path=None):
        a, b = [], []
        edge_attr = []
        node_feature = None
        matrix = np.load(matrix_path)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if(matrix[i][j] > 0):
                    a.append(i)
                    b.append(j)
        edge = [a,b]
        edge_index = torch.tensor(edge, dtype=torch.long)

        return edge_index, edge_attr, node_feature
    
    def get_edge_feature(self, edge_index, x_data):
        pass

    def __getitem__(self, index):
        """
        : data.node_num record the node number of each batch
        : data.x shape is [batch_size, node_num, his_num, message_dim]
        : data.y shape is [batch_size, node_num, pred_num]
        : data.edge_index constructed for torch_geometric
        : data.edge_attr  constructed for torch_geometric
        : data.node_feature shape is [batch_size, node_num, node_dim]
        """

       
        select_dataset = self.data_list[0]
        x_data = self.x_list[select_dataset][index: index+1]
        y_data = self.y_list[select_dataset][index: index+1]
        
        data_i = Data(node_num=1, x=x_data, y=y_data)
        data_i.edge_index = self.edge_index_list[select_dataset]
        data_i.data_name = select_dataset
        A_wave = self.A_list[select_dataset]
        return data_i, A_wave

    
    def __len__(self):
        if self.stage == 'source':
            print("[random permutation] length is decided by training epochs")
            return 100000000
        else:
            data_length = self.x_list[self.data_list[0]].shape[0]
            return data_length