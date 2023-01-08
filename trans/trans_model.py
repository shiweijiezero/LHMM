import os
import sys

base_path = os.path.abspath("../")
sys.path.append(base_path)

import torch.nn as nn
import torch.nn.functional as F
from trans.trans_layer import RGCN
from trans.trans_layer import Attention
import torch
from utils.mygcn_graph import GCN_graph
import pickle
from config import DefaultConfig


class TransModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, rel_names, road_num_nodes, cellular_num_nodes):
        super(TransModel, self).__init__()

        self.gc_layer = RGCN(nfeat, nhid, nhid, rel_names)
        self.attention_layer = Attention(nhid, nhid, nhid)
        self.linear1 = nn.Linear(2 * nhid, nhid)
        self.linear2 = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.linear3 = nn.Linear(1, nhid)
        self.linear4 = nn.Linear(nhid, nhid)
        self.road_embeddings = nn.Embedding(road_num_nodes, DefaultConfig.gcn_nfeat)
        self.cellular_embeddings = nn.Embedding(cellular_num_nodes, DefaultConfig.gcn_nfeat)
        self.get_all_road_embeddings = torch.arange(road_num_nodes).to(DefaultConfig.DEVICE)
        self.get_all_cellular_embeddings = torch.arange(cellular_num_nodes).to(DefaultConfig.DEVICE)
        self.linear5 = nn.Linear(nhid, nhid)
        self.linear6 = nn.Linear(nhid, nhid)
        self.linear7 = nn.Linear(2 * nhid, nhid)

    def forward(
            self,
            dgl_cellular_id_lst,
            label_nids,
            dgl_graph,
            node_distance
    ):
        # 通过异构图GCN获得Embedding
        road_features = self.road_embeddings(self.get_all_road_embeddings)
        road_features = self.linear5(road_features)
        cellular_features = self.cellular_embeddings(self.get_all_cellular_embeddings)
        cellular_features = self.linear6(cellular_features)
        graph_feature_dict = self.gc_layer(dgl_graph, {"road": road_features, "cellular": cellular_features})
        road_embedding = graph_feature_dict['road']
        cellular_embedding = graph_feature_dict['cellular']

        # 选中cellular轨迹周边的road的Embedding
        road_picked_embedding = road_embedding[label_nids]
        # 选中cellular sequence的Embedding
        cellular_sequence_embedding = cellular_embedding[dgl_cellular_id_lst]

        # 获得基站序列对某个道路的attention嵌入表示
        cellular_hidden_embedding = self.attention_layer(
            road_picked_embedding,  # queries
            cellular_sequence_embedding,  # keys
            cellular_sequence_embedding  # values
        )

        # 拼接获得road_cellular_pair
        road_cellular_pair = torch.cat((road_picked_embedding, cellular_hidden_embedding), 1)  # n*2h

        # 获得距离变换得到的Embedding
        node_distance = node_distance.unsqueeze(0).T  # n=>1*n=>n*1
        distance_embedding = self.linear3(node_distance)  # n*nhid
        distance_embedding = F.relu(distance_embedding)
        distance_embedding = self.linear4(distance_embedding)  # n*nhid
        distance_embedding = F.relu(distance_embedding)
        distance_embedding = F.dropout(distance_embedding, self.dropout, training=self.training)

        # 接着做变换，得到输出概率值
        output = self.linear1(road_cellular_pair)  # n*nhid
        output = F.relu(output)
        output = F.dropout(output, self.dropout, training=self.training)

        output = torch.cat((output, distance_embedding), 1)  # 拼接嵌入特征和距离特征 n*2nhid
        output = self.linear7(output)  # n*nhid
        output = F.relu(output)
        output = F.dropout(output, self.dropout, training=self.training)
        output = self.linear2(output)  # n*c
        output = F.softmax(output, dim=1)
        return output

    def get_embedding(
            self,
            road_features,
            cellular_features,
            dgl_cellular_id_lst,
            label_nids,
            dgl_graph
    ):
        # 通过异构图GCN获得Embedding
        road_features = self.road_embeddings(self.get_all_road_embeddings)
        road_features = self.linear5(road_features)
        cellular_features = self.cellular_embeddings(self.get_all_cellular_embeddings)
        cellular_features = self.linear6(cellular_features)
        graph_feature_dict = self.gc_layer(dgl_graph, {"road": road_features, "cellular": cellular_features})
        road_embedding = graph_feature_dict['road']
        cellular_embedding = graph_feature_dict['cellular']

        # 选中cellular轨迹周边的road的Embedding
        road_picked_embedding = road_embedding[label_nids]
        # 选中cellular sequence的Embedding
        cellular_sequence_embedding = cellular_embedding[dgl_cellular_id_lst]
        return road_picked_embedding, cellular_sequence_embedding

    def store_embedding(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # n*h
        x = F.dropout(x, self.dropout, training=False)
        embedding = self.gc2(x, adj)  # n*h
        gcn_graph = GCN_graph()
        G = gcn_graph.gcn_nxgraph
        gcn_node_index_lst = sorted(list(G.nodes()))
        embedding_dict = {}
        for index in range(len(gcn_node_index_lst)):
            node_id = gcn_node_index_lst[index]
            node_embedding = embedding[index]  # 1*h
            embedding_dict[node_id] = node_embedding
        print("保存embedding字典")
        with open(DefaultConfig.embedding_dict_path, mode='wb') as f:
            pickle.dump(embedding_dict, f)
