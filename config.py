import random

import torch
import os
import numpy as np

class DefaultConfig(object):
    nrows = None
    edge_path = 'data/hangzhou/input/map/EdgeInfo.csv'
    # 原始路段文件（WGS84坐标）
    node_path = 'data/hangzhou/input/map/NodeInfo.csv'
    # 原始节点文件（WGS84坐标）
    gps_path = 'data/hangzhou/input/gps/all.csv'
    # 原始GPS文件（GCJ02坐标）
    mee_path = 'data/hangzhou/input/mee/mee.csv'
    # 原始MEE文件（WGS84坐标）

    nxgraph_path = '../data/nxgraph.obj'
    # 路网networkx的object文件
    kdtree_path = "../data/kdtree.obj"
    # 路网节点的kdtree文件
    nid2rids_path = "../data/nid2rids.obj"
    # node_id转road_id字典，其中node_id对应的是所有包含该节点的road_id集合
    pos2nid_path = "../data/pos2nid.obj"
    rid2nids_path = "../data/rid2nids.obj"

    gps_traj_path = "../data/gps_traj.obj"
    # 字典，轨迹id:gps轨迹
    mee_traj_path = "../data/mee_traj.obj"
    # 字典，轨迹id:mee轨迹
    filtered_gps_traj_path = "../data/filtered_gps_traj.obj"
    # 字典，过滤后的轨迹id:gps轨迹
    filtered_mee_traj_path = "../data/filtered_mee_traj.obj"
    # 字典，过滤后的轨迹id:mee轨迹

    # gcn 图结构
    gcn_nxgraph_path = '../data/gcn_nxgraph.obj'
    # 位置转cellular_id字典
    pos2cellular_id_path = '../data/pos2cellular_id.obj'
    # gcn_train_data字典，轨迹id->节点id列表
    gcn_train_data_path = '../data/gcn_train_data.obj'
    # gcn_train_data字典，轨迹id->节点id列表  使用下面这个训练模式！
    gcn_new_train_data_path = '../data/gcn_new_train_data.obj'
    # node2index_dict字典
    node2index_dict_path = '../data/node2index_dict.obj'
    # embedding_dict字典
    road_embedding_path = '../data/road_embedding.obj'
    cell_embedding_path = '../data/cell_embedding.obj'
    # dgl 图
    dgl_graph_path = '../data/dgl_graph.bin'
    pos2dgl_cellular_id_path = '../data/pos2dgl_cellular_id.obj'
    dgl_cellular_id2pos_path='../data/dgl_cellular_id2pos.obj'
    # shortcuts训练数据字典
    shortcuts_train_data_path = '../data/shortcuts_train_data.obj'
    # trans训练数据字典
    trans_train_path='../data/trans_train.obj'
    trans_train_path1='../data/trans_train1.obj'
    trans_train_path2='../data/trans_train2.obj'
    trans_train_path3='../data/trans_train3.obj'
    trans_train_path4='../data/trans_train4.obj'
    trans_train_path5='../data/trans_train5.obj'
    trans_train_path6='../data/trans_train6.obj'
    trans_train_path7='../data/trans_train7.obj'
    trans_train_path8='../data/trans_train8.obj'
    trans_train_path9='../data/trans_train9.obj'
    trans_train_path10='../data/trans_train10.obj'
    dic_cellular_trajectory_path='../data/dic_cellular_trajectory.obj'

    dbsession_path = 'mongodb://192.168.131.191:27017/'
    # MongoDB的路径
    pairedData_mongo_table = "pairedData"
    # 处理后的MEE与GPS对的表
    roadnetwork_e_mongo_table = "roadnetwork_e"
    # 处理后的路网路段表（WGS84坐标）
    roadnetwork_v_mongo_table = "roadnetwork_v"
    # 处理后的路网节点表（WGS84坐标）
    match_result_table_path = "2021-04-08 18:28:12-matchResult"

    output_dir = 'output'
    web_server_port = 5000

    PRECISION = 13
    # 路网中的节点位置坐标精确到的小数点位数，使用四舍五入
    # 由于MongoDB中的已经被四舍五入过了，所以程序中不需要

    roadSpeedFilterPara = 41
    # 过滤掉限速无名路段

    gps_candidate_range = 30
    gps_sigmaM = 10

    DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"  # cuda:1
    # DEVICE = "cpu" # cuda:1

    # GCN part
    use_model="" # 使用模型的话设为模型路径，否则设为空字符串
    # use_model="/home/weijie0311/Project/mapmatching-cikm/data/model/2022-03-26-21-03-128-128-1-1-0.001-0.5-0.8/mlpl-1.pkl" # 使用模型的话设为模型路径，否则设为空字符串
    # use_model="/home/weijie0311/Project/mapmatching-cikm/data/model/2022-03-15-23-02-128-128-1-1-0.001-0.5-0.8/mlpl.pkl" #
    gcn_nfeat = 128  # gcn特征长度
    train_set_rate = 0.8  # 划分训练集
    val_set_rate = 0.2  # 划分验证集 ，实现上是除去训练集的另一部分
    nclass = 2

    seed = 42
    EPOCHS = 2  # 200
    lr = 0.001  # 0.0001
    weight_decay = 1e-4
    hidden = 128 # 128
    dropout = 0.5
    positive_weight = 1  # 正例权重
    negetive_weight = 1
    gcn_mini_data = False

    # trans part
    trans_EPOCHS = 20  # 200
    # trans_use_model = ""
    trans_use_model = "/home/weijie0311/Project/mapmatching-cikm/data/model/trans-2022-03-30-16-00-128-128-0.001-0.5-0.8/mlpl-31.pkl"
    trans_mini_batch = 400
    trans_mini_data = False
    trans_positive_weight = 1  # 正例权重
    trans_negetive_weight = 1


    def __init__(self):
        random.seed(DefaultConfig.seed)
        np.random.seed(DefaultConfig.seed)
        torch.manual_seed(DefaultConfig.seed)
        if (DefaultConfig.DEVICE != "cpu"):
            torch.cuda.manual_seed(DefaultConfig.seed)
