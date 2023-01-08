import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        print(f"this {rel_names}")
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')  # aggregate='mean')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')  # aggregate='mean')

    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class Attention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens):
        super().__init__()
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)  #

    def forward(self, queries, keys, values):
        # queries 首先从a*h变成 a*1*h
        # keys  首先从b*h变成a*b*h
        # values 首先从b*h变成a*b*h
        queries = queries.unsqueeze(1)
        keys = keys.repeat(queries.shape[0], 1, 1)
        values = values.repeat(queries.shape[0], 1, 1)

        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)  # torch.Size([a, 1, 1, h]) torch.Size([a, 1, b, h])
        features = torch.tanh(features)  # 广播相加后为 torch.Size([a, 1, b, h])

        scores = self.w_v(features)  # torch.Size([a, 1, b, 1])
        scores = scores.squeeze(-1)  # w_v消掉最后隐藏层维 torch.Size([a, 1, b])
        # a个queries，b个keys，每个query得到b个score权重

        self.attention_weigths = F.softmax(scores, dim=2)  # 结果取softmax torch.Size([a, 1, b])
        # attention weights和values加权相加
        # 从torch.Size([a, 1, b])与torch.Size([a, b, h])变为torch.Size([a, 1, h])，再去除维度为1的得到torch.Size([a, h])
        return torch.bmm(self.attention_weigths, values).squeeze()
