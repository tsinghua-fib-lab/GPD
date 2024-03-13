import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import time
import sys
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.utils import weight_norm
from torchsummary import summary

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)

        temp = self.conv1(X) + self.conv2(X)
        gate = torch.sigmoid(self.conv3(X))
        out = F.relu(gate * temp)
        # temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        # out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out

class TimeBlock_NonBias(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=4):

        super(TimeBlock_NonBias, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), bias=False)

    def forward(self, X):

        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + self.conv2(X)
        gate = torch.sigmoid(self.conv3(X))
        out = F.relu(gate * temp)
        # temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        # out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out

class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    # def __init__(self, in_channels, spatial_channels, out_channels,
                #  num_nodes):
    def __init__(self, in_channels, spatial_channels, out_channels):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        # self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # print("[Block] X shape is", X.shape)
        t = self.temporal1(X)
        # print("[Block] t1 shape is", t.shape)
        # print("t shape is {}, A_hat shape is {}".format(t.shape, A_hat.shape))
        # print("t shape is {}, A_hat shape is {}".format(t.shape, A_hat.shape))
        # sys.exit(0)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # old: lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # print("lfs shape is {}".format(lfs.shape))
        t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # print("[Block] t2 shape is", t2.shape)
        t3 = self.temporal2(t2)
        # print("[Block] t3 shape is", t3.shape)
        # sys.exit(0)
        # sys.exit(0)
        # return self.batch_norm(t3)
        return t3

class STGCNBlock_NonBias(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    # def __init__(self, in_channels, spatial_channels, out_channels,
                #  num_nodes):
    def __init__(self, in_channels, spatial_channels, out_channels):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock_NonBias, self).__init__()
        self.temporal1 = TimeBlock_NonBias(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock_NonBias(in_channels=spatial_channels,
                                   out_channels=out_channels)
        # self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # print("[Block] X shape is", X.shape)
        t = self.temporal1(X)
        # print("[Block] t1 shape is", t.shape)
        # print("t shape is {}, A_hat shape is {}".format(t.shape, A_hat.shape))
        # print("t shape is {}, A_hat shape is {}".format(t.shape, A_hat.shape))
        # sys.exit(0)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
       
        # print("lfs shape is {}".format(lfs.shape))
        t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # print("[Block] t2 shape is", t2.shape)
        t3 = self.temporal2(t2)
        # print("[Block] t3 shape is", t3.shape)
        # sys.exit(0)
        # sys.exit(0)
        # return self.batch_norm(t3)
        return t3

class STGCN_NonBias(nn.Module):  # WE CHOOSE
    def __init__(self, model_args, task_args):
        super(STGCN_NonBias, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.meta_dim = model_args['meta_dim']
        self.message_dim = model_args['message_dim']
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.build()
    
    def build(self):
        self.block1 = STGCNBlock_NonBias(in_channels=self.message_dim, out_channels=32,
                                 spatial_channels=8)
        self.last_temporal = TimeBlock_NonBias(in_channels=32, out_channels=32)
        self.fully = nn.Linear((self.his_num - 9) * 32,
                               self.pred_num, bias=False)
        self.init_weights()
        # for name, parameter in self.named_parameters():
        #     print(name, ':', parameter.size())
    
    def init_weights(self):
        if isinstance(self.fully, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(self.fully.weight)
        
    
    def forward(self, data, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        X = data.x
        # print("x shape is", X.shape)
        out1 = self.block1(X, A_hat)
        # print("out1 shape is", out1.shape)
        # print("out2 shape is", out2.shape)
        out3 = self.last_temporal(out1)
        # print("out3 shape is", out3.shape)
        
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        # print("out4 shape is", out4.shape)  # out4 shape is torch.Size([64, 4, 6])
        return out4, A_hat

