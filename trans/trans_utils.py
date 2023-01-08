import pickle
import os
import random
import sys

import torch

base_path = os.path.abspath("../")
sys.path.append(base_path)
from config import DefaultConfig
from utils.mygcn_graph import GCN_graph


def load_base_data():
    opt = DefaultConfig()
    gcn_graph = GCN_graph()
    dgl_graph = gcn_graph.dgl_graph
    dic_cellular_traj = None
    with open(opt.dic_cellular_trajectory_path, mode='rb') as f:
        dic_cellular_traj = pickle.load(f)
        print("Reading dic_cellular_traj file...")

    return dgl_graph.to(DefaultConfig.DEVICE), \
           gcn_graph.dgl_cellular_id2pos, \
           dic_cellular_traj


def load_data(flag):
    opt = DefaultConfig()
    file_name = None
    if (flag == 1):
        file_name = opt.trans_train_path1
    if (flag == 2):
        file_name = opt.trans_train_path2
    if (flag == 3):
        file_name = opt.trans_train_path3
    if (flag == 4):
        file_name = opt.trans_train_path4
    if (flag == 5):
        file_name = opt.trans_train_path5
    if (flag == 6):
        file_name = opt.trans_train_path6
    if (flag == 7):
        file_name = opt.trans_train_path7
    if (flag == 8):
        file_name = opt.trans_train_path8
    if (flag == 9):
        file_name = opt.trans_train_path9
    if (flag == 10):
        file_name = opt.trans_train_path10
    trans_train_dic = None
    with open(file_name, mode='rb') as f:
        print("Reading trans_train_dic file...")
        trans_train_dic = pickle.load(f)
    train_data = list(trans_train_dic.items())
    if (DefaultConfig.trans_mini_data == True):
        print("DefaultConfig.trans_mini_batch==True")
        train_data = train_data[:2000]
    random.shuffle(train_data)
    return train_data[:int(DefaultConfig.train_set_rate * len(train_data))], \
           train_data[-int(DefaultConfig.val_set_rate * len(train_data)):]

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
    load_data(1)
