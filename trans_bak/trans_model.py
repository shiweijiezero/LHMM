import os
import sys

from torch.nn.utils.rnn import pack_sequence

from trans_bak.trans_layer import LSTM_Model
from trans_bak.trans_layer import RGCN

base_path = os.path.abspath("../")
sys.path.append(base_path)

import torch.nn as nn
import torch.nn.functional as F
import torch
from config import DefaultConfig


# from pygcn.gcn_model import GCN

class TransModel(nn.Module):
    def __init__(self, nhid, dropout, rel_names, road_num_nodes, cellular_num_nodes):
        super(TransModel, self).__init__()
        self.dropout = dropout
        # self.gcn_model:GCN = torch.load(DefaultConfig.use_model).to(DefaultConfig.DEVICE)  # 导入保存的模型
        self.gc_layer = RGCN(128, nhid, nhid, rel_names)
        self.lstm1 = LSTM_Model(embedding_dim=nhid, hidden_dim=nhid)
        self.lstm2 = LSTM_Model(embedding_dim=nhid, hidden_dim=nhid)
        self.linear1 = nn.Linear(1, nhid)
        self.linear2 = nn.Linear(1, nhid)
        self.linear3 = nn.Linear(2 * nhid, nhid)
        self.linear4 = nn.Linear(nhid, nhid)
        self.linear5 = nn.Linear(2 * nhid, nhid)
        self.linear6 = nn.Linear(nhid, 2)
        self.linear7 = nn.Linear(nhid, nhid)
        self.linear8 = nn.Linear(nhid, nhid)
        self.linear_temp = nn.Linear(nhid, nhid)
        self.road_embeddings = torch.nn.Embedding(road_num_nodes, DefaultConfig.gcn_nfeat)
        self.cellular_embeddings = torch.nn.Embedding(cellular_num_nodes, DefaultConfig.gcn_nfeat)
        self.get_all_road_embeddings = torch.arange(road_num_nodes).to(DefaultConfig.DEVICE)
        self.get_all_cellular_embeddings = torch.arange(cellular_num_nodes).to(DefaultConfig.DEVICE)

    def forward(
            self,
            dgl_cellular_id_lst,
            cellular1_id,
            cellular2_id,
            path_batch_data,  # tuple(path_node)
            dgl_graph,
            cellular_distance,  # batch*1
            path_distance  # batch*1
    ):
        cellular_distance = cellular_distance.to(DefaultConfig.DEVICE)
        path_distance = path_distance.to(DefaultConfig.DEVICE)
        dis=torch.abs(cellular_distance-path_distance)

        # 通过异构图GCN获得Embedding
        # graph_feature_dict=self.gcn_model.gc_layer(dgl_graph, {"road": road_features, "cellular": cellular_features})
        # road_embedding = torch.detach(graph_feature_dict['road'])
        # cellular_embedding = torch.detach(graph_feature_dict['cellular'])

        # 通过异构图GCN获得Embedding
        road_features = self.road_embeddings(self.get_all_road_embeddings)
        road_features = self.linear7(road_features)
        cellular_features = self.cellular_embeddings(self.get_all_cellular_embeddings)
        cellular_features = self.linear8(cellular_features)
        graph_feature_dict = self.gc_layer(dgl_graph, {"road": road_features, "cellular": cellular_features})
        road_embedding = graph_feature_dict['road']
        cellular_embedding = graph_feature_dict['cellular']

        # 选中的cellular embedding
        cellular1_embedding = cellular_embedding[cellular1_id].reshape((1, cellular_embedding.shape[1]))
        cellular2_embedding = cellular_embedding[cellular2_id].reshape((1, cellular_embedding.shape[1]))

        # path的Embedding
        for i in range(len(path_batch_data)):
            path = path_batch_data[i]
            path_embed_seq = road_embedding[path]  # L*hid
            path_batch_data[i] = path_embed_seq
        path_embed = self.lstm1(path_batch_data)

        # 两个cellular点的embedding
        cellular_seq_embedding = torch.cat((cellular1_embedding, cellular2_embedding), dim=0)
        cellular_seq_embed = self.lstm2([cellular_seq_embedding])
        cellular_seq_embed = cellular_seq_embed.repeat(len(path_embed), 1)
        # 得到序列pair特征
        batch_seq_feature = torch.cat((cellular_seq_embed, path_embed), dim=1)

        # 接着做变换，得到输出概率值
        feature1 = self.linear3(batch_seq_feature)  # batch*nhid
        feature1 = F.relu(feature1)
        feature1 = F.dropout(feature1, self.dropout, training=self.training)

        # 获得距离变换得到的Embedding
        batch_dis_embed = self.linear1(dis)  # batch*nid
        batch_dis_embed = F.relu(batch_dis_embed)
        feature2 = self.linear4(batch_dis_embed)  # batch*nhid
        feature2 = F.relu(feature2)
        feature2 = F.dropout(feature2, self.dropout, training=self.training)
        # output = self.linear5(torch.cat((feature1, feature2), dim=1))  # batch*nhid
        output = self.linear_temp(feature1) #######################
        output = F.relu(output)
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.linear6(output)  # batch*2
        output = F.softmax(output, dim=1)
        return output
