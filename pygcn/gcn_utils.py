import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy import sparse

import networkx as nx
import os
import sys

base_path = os.path.abspath("../")
sys.path.append(base_path)
from utils.mygcn_graph import GCN_graph
from config import DefaultConfig
from utils.mymee import MEEDataset


# 导入数据


def lap_norm(adj):
    # 增加自循环且根据度进行拉普拉斯归一化
    adj += sparse.eye(adj.shape[0])  # 为每个结点增加自环
    degree = np.array(adj.sum(1))  # 为每个结点计算度
    degree = sparse.diags(np.power(degree, -0.5).flatten())
    return degree.dot(adj).dot(degree)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def one_hot(adj, node_idx_lst):
    labels_one_hot = np.zeros((adj.shape[0], 2))
    labels_one_hot[:, 0] = 1
    labels_one_hot[node_idx_lst, 0] = 0
    labels_one_hot[node_idx_lst, 1] = 1
    return labels_one_hot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 这里可以用max或者sum
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def load_data():
    # 在dgl图中，road id就是node index，cellular id 也是cellular index
    # 不再像networkx中需要索引查找
    gcn_graph = GCN_graph()
    dgl_graph = gcn_graph.dgl_graph
    gcn_train_data = gcn_graph.gcn_train_data
    train_data = list(gcn_train_data.items())
    if(DefaultConfig.gcn_mini_data==True):
        print("DefaultConfig.gcn_mini_data==True")
        train_data=train_data[:2000]
    random.shuffle(train_data)
    return train_data[:int(DefaultConfig.train_set_rate * len(train_data))], \
           train_data[-int(DefaultConfig.val_set_rate * len(train_data)):], \
           dgl_graph.to(DefaultConfig.DEVICE), \
           gcn_graph.dgl_cellular_id2pos


def load_data_old():
    """
    discard!
    """
    gcn_graph = GCN_graph()

    G = gcn_graph.gcn_nxgraph
    # print(G.number_of_nodes()) # 309481
    # print(G.number_of_edges()) # 1656216

    # G=MapData.nxgraph
    # print(G.number_of_nodes()) # 294982
    # print(G.number_of_edges()) # 554077
    MEE_Data = MEEDataset()

    # 可以按照下标获取对应的node节点
    gcn_node_index_lst = sorted(list(G.nodes()))
    # gcn_node_index_lst=sorted(list(G.nodes()))[:100]  # 测试使用，避免占用内存过大
    node2index_dict = gcn_graph.node2index_dict
    # 按照 node的大小顺序来转换为邻接矩阵
    adj = nx.adjacency_matrix(G, nodelist=gcn_node_index_lst, weight="gcn_weight")
    adj = lap_norm(adj)
    embeddings = torch.nn.Embedding(G.number_of_nodes(), DefaultConfig.gcn_nfeat).to(DefaultConfig.DEVICE)

    # features = torch.rand(G.number_of_nodes(), DefaultConfig.gcn_nfeat)
    # 归一化
    # adj=normalize(adj)
    # features=torch.tensor(normalize(features))
    # adj=F.normalize(adj)

    train_data_dict = gcn_graph.gcn_train_data
    train_set1 = []
    train_set2 = []
    for traj_id, node_id_lst in train_data_dict.items():
        mee_pos_lst = MEE_Data.get_mee_line_without_ts(traj_id)
        mee_idx_lst = [node2index_dict[gcn_graph.pos2cellular_id[pos]] for pos in mee_pos_lst]
        match_node_idx_lst = [node2index_dict[node_id] for node_id in node_id_lst]
        # 普通标签
        labels = np.zeros(adj.shape[0])
        labels[match_node_idx_lst] = 1
        # one-hot标签
        # labels=one_hot(adj,node_idx_lst)
        labels = torch.tensor(labels, dtype=torch.int8).to(DefaultConfig.DEVICE)
        mee_idx_lst = torch.tensor(mee_idx_lst, dtype=torch.long).to(DefaultConfig.DEVICE)
        train_set1.append(mee_idx_lst)
        train_set2.append(labels)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = adj.to(DefaultConfig.DEVICE)
    return adj, \
           embeddings, \
           train_set1[:int(DefaultConfig.train_set_rate * len(train_set1))], \
           train_set2[:int(DefaultConfig.train_set_rate * len(train_set2))], \
           train_set1[-int(DefaultConfig.val_set_rate * len(train_set1)):], \
           train_set2[-int(DefaultConfig.val_set_rate * len(train_set2)):]


def accuracy(output, labels):
    preds = output.argmax(dim=1)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def recall_percent(output, labels):
    preds = output.max(1)[1].type_as(labels)  # [1]是取列下标，[0]是取值
    labels_index_for_1 = torch.where(labels == 1)[0]
    preds_value_for_1 = preds[labels_index_for_1]
    all_num = labels_index_for_1.shape[0]
    recall_num = torch.count_nonzero(preds_value_for_1).item()
    return recall_num / all_num


if __name__ == "__main__":
    load_data()
