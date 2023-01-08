import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence


class LSTM_Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,num_layers=3,batch_first=True)


    def forward(self, path_lst):
        path_data=pack_sequence(path_lst,enforce_sorted=False)
        # path: batch*length*embedding，batch用列表，length用tensor，embedding用tensor
        out, hidden = self.lstm(path_data)
        last_layer=hidden[0][-1]  # 0 是为了拿到hidden输出舍弃memory输出，-1 是为了拿到多层rnn hidden的最上面一层，得到batch*embedding大小
        return last_layer

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