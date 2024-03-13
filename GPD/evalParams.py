from PredictionModel.NetSet import StgnnSet
import sys
from PredictionModel.datasets import *
from PredictionModel.utils import *
from tqdm import tqdm


def getNewIndex(nodeindex, addr):
    A = np.load(addr)
    nodeset = []
    for i in range(A.shape[0]):
        if A[i][nodeindex] != 0  or  A[nodeindex][i] != 0:
            nodeset.append(i)
    # nodeset = nodeset2
    for i in range(len(nodeset)):
        if nodeset[i]==nodeindex:
            return i


def evalParams(params, config, device, logger, targetDataset, writer, epoch, basemodel):
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    model = StgnnSet(data_args, task_args, model_args, basemodel).to(device=device)
    
    node_num = 0
    epochs = 0
    if targetDataset == 'None':
        datasetsall = ['DC','BM','man']
    elif targetDataset == 'TrafficNone':
        datasetsall = ['metr-la', 'pems-bay', 'chengdu_m', 'shenzhen']
    else:
        datasetsall = [targetDataset]
        
    results = []
    for dataset in datasetsall:
        if dataset == 'DC':
            node_num = 194
            locbias = 0
        elif dataset == 'BM':
            node_num = 267
            locbias = 194 if targetDataset=='None' else 0
        elif dataset == 'man':
            node_num = 195
            locbias = 461 if targetDataset=='None' else 0
            
        elif dataset == 'metr-la':
            node_num = 207
            locbias = 0
        elif dataset == 'pems-bay':
            node_num = 325
            locbias = 207 if targetDataset=='TrafficNone' else 0
        elif dataset == 'shenzhen':
            node_num = 627
            locbias = 532 if targetDataset=='TrafficNone' else 0
        elif dataset == 'chengdu_m':
            node_num = 524
            locbias = 1159 if targetDataset=='TrafficNone' else 0

        step = 0
        params = params.reshape((params.shape[0],-1))
        for node_index in tqdm(range(node_num)):
            model = StgnnSet(data_args, task_args, model_args, model=basemodel).to(device=device)
            
            test_dataset = traffic_dataset2(data_args, task_args, node_index, "test", False, dataset)
            test_meanstd = [test_dataset.mean, test_dataset.std]
            test_dataloader = DataLoader(test_dataset, batch_size=task_args['test_batch_size'], shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
            AAddr = data_args[dataset]['adjacency_matrix_path']
            
            newindex = getNewIndex(node_index,AAddr)
            
            param = params[node_index + locbias]
            outputs, y_label = model.task4eval(param, newindex, node_index, test_dataloader, logger, test_meanstd, basemodel) 
            
            if step == 0 :
                out = outputs
                truth = y_label
                step = 1
            else:
                out = np.concatenate((out, outputs),axis=2)
                truth = np.concatenate((truth, y_label),axis=2)

        result = metric_func(pred=out, y=truth, times=task_args['pred_num'])
        results.append(result)

        if dataset == 'DC':
            writer.add_scalar('DC_mae', np.mean(result['MAE']), epoch)
            writer.add_scalar('DC_rmse', np.mean(result['RMSE']), epoch)
        elif dataset == 'BM':
            writer.add_scalar('BM_mae', np.mean(result['MAE']), epoch)
            writer.add_scalar('BM_rmse', np.mean(result['RMSE']), epoch)
        elif dataset == 'man':
            writer.add_scalar('man_mae', np.mean(result['MAE']), epoch)
            writer.add_scalar('man_rmse', np.mean(result['RMSE']), epoch)

    metricsum = []
    for result in results:
        metricsum.append(result_print(result, logger,info_name='Evaluate'))
        
    return sum(metricsum)
        
        
        
        