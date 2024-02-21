import sys
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch
from Models.meta_gwn import *
from Models.meta_stgcn import *
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from utils import *


def asym_adj(adj):
    adj = adj.cpu().numpy()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

class StgnnSet(nn.Module): 
    """
    MAML-based Few-shot learning architecture for STGNN
    """
    def __init__(self, data_args, task_args, model_args, model='GRU', node_num=207):
        super(StgnnSet, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.model_args = model_args
        
        self.update_lr = model_args['update_lr']
        self.meta_lr = model_args['meta_lr']
        self.update_step = model_args['update_step']
        self.update_step_test = model_args['update_step_test']
        self.task_num = task_args['task_num']
        self.model_name = model

        self.loss_lambda = model_args['loss_lambda']
        print("loss_lambda = ", self.loss_lambda)

        if model == 'v_STGCN5':  # STGCN 5.0 which we choose
            self.model = STGCN_NonBias(model_args, task_args)
        elif model == 'v_GWN':
            self.model = v_GWN()  

        print("model params: ", count_parameters(self.model))

        # indexstart = [0]
        # lastindex = 0
        # shapes = []

        # print("model num of net", len(list(self.model.children())))
        # for index ,(name, param) in enumerate(self.model.named_parameters()):
        #     print(str(index) + ' '+ name + ':', param.size())
        #     size = tuple(param.size())
        #     shapes.append(size)
        #     length = 1
        #     for i in size:
        #         length = length*i
        #     nowindex = lastindex + length
        #     lastindex = nowindex
        #     indexstart.append(nowindex)
                            
        # print(indexstart)
        # print(shapes)
        # print("model params: ", count_parameters(self.model))
        # sys.exit(0)

        self.meta_optim = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)

        # self.meta_optim = torch.optim.SGD(self.model.parameters(), lr=self.update_lr, momentum=0.9)
        self.loss_criterion = nn.MSELoss()


    def graph_reconstruction_loss(self, meta_graph, adj_graph):   
        adj_graph = adj_graph.unsqueeze(0).float()
        for i in range(meta_graph.shape[0]):
            if i == 0:
                matrix = adj_graph
            else:
                matrix = torch.cat((matrix, adj_graph), 0)
        criteria = nn.MSELoss()
        loss = criteria(meta_graph, matrix.float())
        return loss
    

    def calculate_loss(self, out, y, meta_graph, matrix, stage='target', graph_loss=True, loss_lambda=1):
        if loss_lambda == 0:
            loss = self.loss_criterion(out, y)
        if graph_loss:
            if stage == 'source' or stage == 'target_maml':
                loss_predict = self.loss_criterion(out, y)  
                loss_reconsturct = self.graph_reconstruction_loss(meta_graph, matrix)
            else:
                loss_predict = self.loss_criterion(out, y)
                loss_reconsturct = self.loss_criterion(meta_graph, matrix.float())
            loss = loss_predict + loss_lambda * loss_reconsturct
        else:
            loss = self.loss_criterion(out, y)

        return loss
    

    def forward(self, data, matrix):
        out, meta_graph = self.model(data, matrix)
        return out, meta_graph

    def finetuning(self, target_dataloader, test_dataloader, target_epochs,logger):
        """
        finetunning stage in MAML
        """
        maml_model = deepcopy(self.model)

        optimizer = optim.Adam(maml_model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        min_MAE = 10000000
        best_result = ''
        best_meta_graph = -1

        for epoch in tqdm(range(target_epochs)):
            train_losses = []
            start_time = time.time()
            maml_model.train()

            for step, (data, A_wave) in enumerate(target_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]

                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out, meta_graph = maml_model(data, adj_mx)
                else:
                    out, meta_graph = maml_model(data, A_wave[0].float())
                if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN']:
                    loss = self.loss_criterion(out, data.y)
                else:
                    # loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', graph_loss=False)
                    loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', loss_lambda=self.loss_lambda)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().numpy())
            avg_train_loss = sum(train_losses)/len(train_losses)
            end_time = time.time()
            if epoch % 5 == 0:
                logger.info("[Target Fine-tune] epoch #{}/{}: loss is {}, fine-tuning time is {}".format(epoch+1, target_epochs, avg_train_loss, end_time-start_time))

        with torch.no_grad():
            test_start = time.time() 
            maml_model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out, meta_graph = maml_model(data, adj_mx)
                else:
                    out, meta_graph = maml_model(data, A_wave[0].float())

                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()

            result_print(result, logger,info_name='Evaluate')
            logger.info("[Target Test] testing time is {}".format(test_end-test_start))

    def taskTrain(self, taskmode, target_dataloader, test_dataloader, target_epochs, logger, meanstd):

        optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        min_MAE = 10000000
        best_result = ''
        best_meta_graph = -1

        for epoch in tqdm(range(target_epochs)):
            train_losses = []
            start_time = time.time()
            self.model.train()

            for step, (data, A_wave) in enumerate(target_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]

                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out, meta_graph = self.model(data, adj_mx)
                else:
                    out, meta_graph = self.model(data, A_wave[0].float())

                out = out * meanstd[1] + meanstd[0]
                gold = data.y * meanstd[1] + meanstd[0]

                # print(out.shape)
                # print(data.y.shape)
                # sys.exit(0)
                loss = self.loss_criterion(out, gold)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().numpy())
            avg_train_loss = sum(train_losses)/len(train_losses)
            end_time = time.time()
            if epoch % 5 == 0:
                logger.info("[Target Fine-tune] epoch #{}/{}: loss is {}".format(epoch+1, target_epochs, avg_train_loss))
                if taskmode=='task3':
                    torch.save(self.model,'Param/{}_inside.pt'.format(taskmode))
        
        # torch.save(self.model,'Param/{}_inside.pt'.format(taskmode))

        with torch.no_grad():
            self.model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out, meta_graph = self.model(data, adj_mx)
                else:
                    out, meta_graph = self.model(data, A_wave[0].float())
                
                out = out * meanstd[1] + meanstd[0]
                gold = data.y * meanstd[1] + meanstd[0]

                if step == 0:
                    outputs = out
                    y_label = gold
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, gold))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])

            result_print(result, logger,info_name='Evaluate')


    
    def node_taskTrain(self, modelname, node_index, init_index, target_dataloader, test_dataloader, train_meanstd, test_meanstd, target_epochs,logger, test_dataset, ifnewname, aftername):

        if ifnewname == 1:
            rootpath = 'Param/Task4/{}_{}_{}/' .format(test_dataset, modelname, aftername)
        else:
            rootpath = 'Param/Task4/{}_{}/' .format(test_dataset, modelname)
        folder = os.path.exists(rootpath + 'Node_{}'.format(init_index))
        if not folder:
            os.makedirs(rootpath + 'Node_{}' .format(init_index))
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        mintestloss = 9999999
        bestmodel = deepcopy(self.model)
        badtestres = 0
        
        for epoch in tqdm(range(target_epochs)):
            train_losses = []
            self.model.train()

            for step, (data, A_wave) in enumerate(target_dataloader):
                # print('data.x.shape')
                # print(data.x.shape)
                # sys.exit(0)
            
                data, A_wave = data.cuda(), A_wave.cuda()

                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out = self.model(data, adj_mx)
                else:
                    out, _ = self.model(data, A_wave[0].float())

                outputs = out[:,node_index,:]*train_meanstd[1]+train_meanstd[0]
                y_label = data.y[:,node_index,:]*train_meanstd[1]+train_meanstd[0]

                 
                loss = self.loss_criterion(outputs, y_label) 
                #  [:,node_index,:].unsqueeze(1)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().numpy())
            avg_train_loss = sum(train_losses)/len(train_losses)

            if epoch % 5 == 0:
                with torch.no_grad():  
                    self.model.eval()
                    
                    test_losses = []
                    for step, (data, A_wave) in enumerate(test_dataloader):
                                        
                        data, A_wave = data.cuda(), A_wave.cuda()
                        if self.model_name == 'v_GWN':
                            adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                            out = self.model(data, adj_mx)
                        else:
                            out, meta_graph = self.model(data, A_wave[0].float())

                        # prediction = rescale(out[:,node_index,:], test_meanstd[0], test_meanstd[1])
                        # truth = rescale(data.y[:,node_index,:], test_meanstd[0], test_meanstd[1]) 
                        out = out[:,node_index,:]*train_meanstd[1]+train_meanstd[0]
                        datay = data.y[:,node_index,:]*train_meanstd[1]+train_meanstd[0]

                        # loss = self.loss_criterion(out[:,node_index,:], data.y[:,node_index,:])
                        loss = self.loss_criterion(out, datay)
                        test_losses.append(loss.detach().cpu().numpy())

                        # if step == 0:
                        #     outputs = out[:,node_index,:].unsqueeze(1)  # shape: [128 1 6]
                        #     y_label = data.y[:,node_index,:].unsqueeze(1)
                        # else:
                        #     outputs = torch.cat((outputs, out[:,node_index,:].unsqueeze(1)),dim=1)
                        #     y_label = torch.cat((y_label, data.y[:,node_index,:].unsqueeze(1)),dim=1)
                        if step == 0:
                            outputs = out.unsqueeze(1)  # shape: [128 1 6]
                            y_label = datay.unsqueeze(1)
                        else:
                            outputs = torch.cat((outputs, out.unsqueeze(1)),dim=1)
                            y_label = torch.cat((y_label, datay.unsqueeze(1)),dim=1)
                
                    avg_test_loss = sum(test_losses)/len(test_losses)
                    # logger.info("[Node-level-test] epoch #{}: test loss is {}".format(epoch, avg_test_loss))

                    outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
                    y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
                    # outputs = rescale(outputs, test_meanstd[0], test_meanstd[1])
                    # y_label = rescale(y_label, test_meanstd[0], test_meanstd[1])  
                    result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num']) 

                    # result_print(result, logger,info_name='Evaluate') 
                    
                    if avg_test_loss<mintestloss:
                        bestmodel = deepcopy(self.model)
                        mintestloss = avg_test_loss
                        badtestres = 0
                    else:
                        badtestres = badtestres + 1
                        if badtestres == 100:  
                            break
                    
                    total_RMSE, total_MAE = result['RMSE'], result['MAE']
                    lossmetric = [epoch, avg_train_loss, avg_test_loss, 
                                total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5],
                                total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]]
                    with open(rootpath + 'Node_{}/NodeTrain_{}_LossAndMetric.txt' .format(init_index, init_index), 'a') as f:
                        f.write(" ".join(str(i) for i in lossmetric) + '\n')
                            
                            
        torch.save(bestmodel,rootpath + 'task4_{}.pt' .format(init_index))

        with torch.no_grad():
            self.model = torch.load(rootpath + 'task4_{}.pt' .format(init_index))
            self.model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):

                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape

                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out = self.model(data, adj_mx)
                else:
                    out, meta_graph = self.model(data, A_wave[0].float())
                
                out = out[:,node_index,:]*train_meanstd[1]+train_meanstd[0]
                datay = data.y[:,node_index,:]*train_meanstd[1]+train_meanstd[0]
                
                # print(out.shape)
                # print(data.y.shape)
                # sys.exit(0)
                # if step == 0:
                #     outputs = out[:,node_index,:].unsqueeze(1)
                #     y_label = data.y[:,node_index,:].unsqueeze(1)
                # else:
                #     outputs = torch.cat((outputs, out[:,node_index,:].unsqueeze(1)),dim=1)
                #     y_label = torch.cat((y_label, data.y[:,node_index,:].unsqueeze(1)),dim=1)
                if step == 0:
                    outputs = out.unsqueeze(1)  # shape: [128 1 6]
                    y_label = datay.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, out.unsqueeze(1)),dim=1)
                    y_label = torch.cat((y_label, datay.unsqueeze(1)),dim=1)
            # print(outputs.shape)
            # print(y_label.shape)

            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])

            result_print(result, logger,info_name='Evaluate')
            # print(type(outputs))
            # print(outputs.shape)
            return outputs, y_label
    
    def node_eval(self, node_index, test_dataloader, logger, test_dataset):

        with torch.no_grad():
            self.model = torch.load('Param/Task3_1/{}/task3_{}.pt' .format(test_dataset, node_index))
            test_start = time.time() 
            self.model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):

                A_wave = A_wave[0][node_index,:].unsqueeze(0)
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape

                if self.model_name == 'GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out, meta_graph = self.model(data, adj_mx)
                else:
                    out, meta_graph = self.model(data, A_wave.float())

                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()

            result_print(result, logger,info_name='Evaluate')
            logger.info("[Test] testing time is {}".format(test_end-test_start))
            # print(type(outputs))
            # print(outputs.shape)
            return outputs, y_label
    
    def taskEval(self, test_dataloader, logger):
        self.model = torch.load('Param/task1.pt')
        with torch.no_grad():
            test_start = time.time() 
            self.model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape

                if self.model_name == 'GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out, meta_graph = self.model(data, adj_mx)
                else:
                    out, meta_graph = self.model(data, A_wave[0].float())

                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            # print(outputs.shape)
            # print(y_label.shape)
            # sys.exit(0)
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()

            result_print(result, logger,info_name='Evaluate')
            logger.info("[Target Test] testing time is {}".format(test_end-test_start))


    def task5(self, taskmode, target_dataloader, test_dataloader, target_epochs,logger):

        optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        min_MAE = 10000000
        best_result = ''
        best_meta_graph = -1

        for epoch in tqdm(range(target_epochs)):
            train_losses = []
            start_time = time.time()
            self.model.train()

            for step, (data, A_wave) in enumerate(target_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]

                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out, meta_graph = self.model(data, adj_mx)
                else:
                    out, meta_graph = self.model(data, A_wave[0].float())

                if self.model_name in ['v_GRU', 'r_GRU', 'v_STGCN', 'Node_STGCN']:
                    loss = self.loss_criterion(out, data.y)
                else:
                    # loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', graph_loss=False)
                    loss = self.calculate_loss(out, data.y, meta_graph, A_wave, 'test', loss_lambda=self.loss_lambda)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().numpy())
            avg_train_loss = sum(train_losses)/len(train_losses)
            end_time = time.time()
            if epoch % 5 == 0:
                logger.info("[Target Fine-tune] epoch #{}/{}: loss is {}, fine-tuning time is {}".format(epoch+1, target_epochs, avg_train_loss, end_time-start_time))
                if taskmode=='task3':
                    torch.save(self.model,'Param/{}_inside.pt'.format(taskmode))
        
        torch.save(self.model,'Param/{}_inside.pt'.format(taskmode))

        with torch.no_grad():
            test_start = time.time() 
            self.model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):
                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape
                hidden = torch.zeros(batch_size, node_num, self.model_args['hidden_dim']).cuda()

                if self.model_name == 'GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out, meta_graph = self.model(data, adj_mx)
                else:
                    out, meta_graph = self.model(data, A_wave[0].float())

                if step == 0:
                    outputs = out
                    y_label = data.y
                else:
                    outputs = torch.cat((outputs, out))
                    y_label = torch.cat((y_label, data.y))
            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])
            test_end = time.time()

            result_print(result, logger,info_name='Evaluate')
            logger.info("[Target Test] testing time is {}".format(test_end-test_start))

    def node_taskTrain2(self, param, modelname, node_index, init_index, target_dataloader, test_dataloader, train_meanstd, test_meanstd, target_epochs,logger, test_dataset, ifnewname, aftername):

        if modelname == 'v_STGCN5':
            indexstart = [0, 256, 512, 768, 1024, 2048, 3072, 4096, 8192, 12288, 16384, 16960]
            shapes = [(32, 8), (32, 2, 1, 4), (32, 2, 1, 4), (32, 2, 1, 4), (32, 8, 1, 4), 
                    (32, 8, 1, 4), (32, 8, 1, 4), (32, 32, 1, 4), (32, 32, 1, 4), 
                    (32, 32, 1, 4), (6, 96)]
        else:
            # indexstart = [0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 
            #               2816, 3072, 3328, 3584, 3840, 4096, 4224, 4352, 4480, 4608, 
            #               4736, 4864, 4992, 5120, 5248, 5376, 5504, 5632, 5760, 5888, 
            #               6016, 6144, 6160, 6176, 6192, 6208, 6224, 6240, 6256, 6272, 
            #               6288, 6304, 6320, 6336, 6352, 6368, 6384, 6400, 7040, 7680, 
            #               8320, 8960, 9600, 10240, 10880, 11520, 11552, 11808, 11904]
            # shapes = [(8, 16, 1, 2), (8, 16, 1, 2), (8, 16, 1, 2), (8, 16, 1, 2), 
            #           (8, 16, 1, 2), (8, 16, 1, 2), (8, 16, 1, 2), (8, 16, 1, 2), 
            #           (8, 16, 1, 2), (8, 16, 1, 2), (8, 16, 1, 2), (8, 16, 1, 2), 
            #           (8, 16, 1, 2), (8, 16, 1, 2), (8, 16, 1, 2), (8, 16, 1, 2), 
            #           (16, 8, 1, 1), (16, 8, 1, 1), (16, 8, 1, 1), (16, 8, 1, 1), 
            #           (16, 8, 1, 1), (16, 8, 1, 1), (16, 8, 1, 1), (16, 8, 1, 1), 
            #           (16, 8, 1, 1), (16, 8, 1, 1), (16, 8, 1, 1), (16, 8, 1, 1), 
            #           (16, 8, 1, 1), (16, 8, 1, 1), (16, 8, 1, 1), (16, 8, 1, 1), 
            #           (16,), (16,), (16,), (16,), (16,), (16,), (16,), (16,), 
            #           (16,), (16,), (16,), (16,), (16,), (16,), (16,), (16,), 
            #           (16, 40, 1, 1), (16, 40, 1, 1), (16, 40, 1, 1), (16, 40, 1, 1), 
            #           (16, 40, 1, 1), (16, 40, 1, 1), (16, 40, 1, 1), (16, 40, 1, 1), 
            #           (16, 2, 1, 1), (16, 16, 1, 1), (6, 16, 1, 1)]      
            indexstart = [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 
                          18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768, 
                          33792, 34816, 35840, 36864, 37888, 38912, 39936, 40960, 
                          41984, 43008, 44032, 45056, 46080, 47104, 48128, 49152, 
                          49184, 49216, 49248, 49280, 49312, 49344, 49376, 49408, 
                          49440, 49472, 49504, 49536, 49568, 49600, 49632, 49664, 
                          54784, 59904, 65024, 70144, 75264, 80384, 85504, 90624, 
                          90688, 91712, 91904]
            shapes = [(32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), 
                      (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), 
                      (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), 
                      (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), (32, 32, 1, 2), 
                      (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), 
                      (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), 
                      (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), 
                      (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), (32, 32, 1, 1), 
                      (32,), (32,), (32,), (32,), (32,), (32,), (32,), (32,), 
                      (32,), (32,), (32,), (32,), (32,), (32,), (32,), (32,), 
                      (32, 160, 1, 1), (32, 160, 1, 1), (32, 160, 1, 1), (32, 160, 1, 1), 
                      (32, 160, 1, 1), (32, 160, 1, 1), (32, 160, 1, 1), (32, 160, 1, 1), 
                      (32, 2, 1, 1), (32, 32, 1, 1), (6, 32, 1, 1)]
        index = 0
        for key in self.model.state_dict().keys():
            # print(key)
            if ('running_mean' not in key) and ('running_var' not in key) and ('num_batches_tracked' not in key):
                pa = torch.tensor(param[indexstart[index]:indexstart[index+1]])
                pa = torch.reshape(pa, shapes[index])                
                self.model.state_dict()[key].copy_(pa)               
                index = index + 1
        print('params load ok')
        
        if ifnewname == 1:
            rootpath = 'Param/Task7/{}_{}_{}/' .format(test_dataset, modelname, aftername)
        else:
            rootpath = 'Param/Task7/{}_{}/' .format(test_dataset, modelname)
        folder = os.path.exists(rootpath + 'Node_{}'.format(init_index))
        if not folder:
            os.makedirs(rootpath + 'Node_{}' .format(init_index)) 
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        mintestloss = 9999999
        bestmodel = deepcopy(self.model)
        badtestres = 0

        for epoch in tqdm(range(target_epochs)):
            train_losses = []
            self.model.train()

            for step, (data, A_wave) in enumerate(target_dataloader):
            
                data, A_wave = data.cuda(), A_wave.cuda()

                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out = self.model(data, adj_mx)
                else:
                    out, _ = self.model(data, A_wave[0].float())
                
                outputs = out[:,node_index,:]*train_meanstd[1]+train_meanstd[0]
                y_label = data.y[:,node_index,:]*train_meanstd[1]+train_meanstd[0]
                loss = self.loss_criterion(outputs, y_label) 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.detach().cpu().numpy())
            avg_train_loss = sum(train_losses)/len(train_losses)

            if epoch % 5 == 0:
                with torch.no_grad():  
                    self.model.eval()
                    
                    test_losses = []
                    for step, (data, A_wave) in enumerate(test_dataloader):
                                        
                        data, A_wave = data.cuda(), A_wave.cuda()
                        if self.model_name == 'v_GWN':
                            adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                            out = self.model(data, adj_mx)
                        else:
                            out, meta_graph = self.model(data, A_wave[0].float())

                        out = out[:,node_index,:]*train_meanstd[1]+train_meanstd[0]
                        datay = data.y[:,node_index,:]*train_meanstd[1]+train_meanstd[0]

                        loss = self.loss_criterion(out, datay)
                        test_losses.append(loss.detach().cpu().numpy())

                        if step == 0:
                            outputs = out.unsqueeze(1)  # shape: [128 1 6]
                            y_label = datay.unsqueeze(1)
                        else:
                            outputs = torch.cat((outputs, out.unsqueeze(1)),dim=1)
                            y_label = torch.cat((y_label, datay.unsqueeze(1)),dim=1)
                
                    avg_test_loss = sum(test_losses)/len(test_losses)
                    
                    outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
                    y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
                    
                    result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num']) 

                    
                    if avg_test_loss<mintestloss:
                        bestmodel = deepcopy(self.model)
                        mintestloss = avg_test_loss
                        badtestres = 0
                    else:
                        badtestres = badtestres + 1
                        if badtestres == 100:  
                            break
                    
                    total_RMSE, total_MAE = result['RMSE'], result['MAE']
                    lossmetric = [epoch, avg_train_loss, avg_test_loss, 
                                total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5],
                                total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]]

                    with open(rootpath + 'Node_{}/NodeTrain_{}_LossAndMetric.txt' .format(init_index, init_index), 'a') as f:
                        f.write(" ".join(str(i) for i in lossmetric) + '\n')
                            
                            
        # torch.save(bestmodel,rootpath + 'task7_{}.pt' .format(init_index)) 

        with torch.no_grad():
            # self.model = torch.load(rootpath + 'task7_{}.pt' .format(init_index))
            self.model = bestmodel
            self.model.eval()
            for step, (data, A_wave) in enumerate(test_dataloader):

                data, A_wave = data.cuda(), A_wave.cuda()
                data.node_num = data.node_num[0]
                batch_size, node_num, seq_len, _ = data.x.shape

                if self.model_name == 'v_GWN':
                    adj_mx = [A_wave[0].float(), (A_wave[0].float()).t()]
                    out = self.model(data, adj_mx)
                else:
                    out, meta_graph = self.model(data, A_wave[0].float())
                
                out = out[:,node_index,:]*train_meanstd[1]+train_meanstd[0]
                datay = data.y[:,node_index,:]*train_meanstd[1]+train_meanstd[0]
                
                if step == 0:
                    outputs = out.unsqueeze(1) 
                    y_label = datay.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, out.unsqueeze(1)),dim=1)
                    y_label = torch.cat((y_label, datay.unsqueeze(1)),dim=1)

            outputs = outputs.permute(0, 2, 1).detach().cpu().numpy()
            y_label = y_label.permute(0, 2, 1).detach().cpu().numpy()
            result = metric_func(pred=outputs, y=y_label, times=self.task_args['pred_num'])

            result_print(result, logger,info_name='Evaluate')
            return outputs, y_label
