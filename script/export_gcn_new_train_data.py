import os
import pickle
import random
import sys

base_path = os.path.abspath("../")
sys.path.append(base_path)
from config import DefaultConfig
from utils import mydb
import networkx
from tqdm import tqdm
import pickle
from utils import mymap
from scipy.spatial.kdtree import KDTree
import numpy as np
import torch
import dgl

opt = DefaultConfig()


def export_networkx_obj():
    MapData = mymap.MapDataset()

    shortcuts_train_data = {}
    pos2cellular_id = {}
    pos2dgl_cellular_id = {}

    mongo_db = mydb.MYDB().db
    match_result_table = mongo_db[opt.match_result_table_path]
    match_result_lines = match_result_table.find()

    max_id = max(MapData.nxgraph.nodes)
    counter = max_id + 1
    # networkx的基站id是从road id后面续着的
    dgl_cellular_counter = 0
    # dgl图的基站id是从0开始的，独立的
    print(f"counter:{counter}")
    for line in tqdm(match_result_lines):
        traj_id = line["id"]

        filtered_traj_mee_string = line["filtered_mee"]
        filtered_traj_mee_lst = filtered_traj_mee_string.split("|")[1].split(",")

        for traj in filtered_traj_mee_lst:
            lng, lat = traj.split()
            lng = float(lng)
            lat = float(lat)
            if ((lng, lat) not in pos2cellular_id):
                pos2cellular_id[(lng, lat)] = counter
                counter += 1
            if ((lng, lat) not in pos2dgl_cellular_id):
                pos2dgl_cellular_id[(lng, lat)] = dgl_cellular_counter
                dgl_cellular_counter += 1

        # 可以让MEE点与所有MEE匹配路径点形成边(这里可以是MEE匹配结果，也可以是GPS匹配结果)
        gps_matched_routes_str = line["gpsmatch_routes"]
        gps_matched_routes_lst = gps_matched_routes_str.split(",")
        dgl_cellular_id_lst = []
        train_data=[]
        for i in range(len(filtered_traj_mee_lst)):
            traj = filtered_traj_mee_lst[i]
            mee_lng, mee_lat = traj.split()
            mee_pos = (float(mee_lng), float(mee_lat))

            dgl_cellular_id = pos2dgl_cellular_id[mee_pos]
            dgl_cellular_id_lst.append(dgl_cellular_id)

            temp1 = []
            for j in range(len(gps_matched_routes_lst)):
                matched_road = gps_matched_routes_lst[j]
                matched_nodes = MapData.rid2nids[matched_road]
                node_src, node_dst = matched_nodes[0], matched_nodes[1]
                temp1.append(node_src)
                temp1.append(node_dst)
            # 去重复
            positive_nids = []
            negetive_nids = []
            temp = list(set(temp1))
            temp.sort(
                key=lambda x: MapData.get_distance(
                    (MapData.nxgraph.nodes[x]["lng"], MapData.nxgraph.nodes[x]["lat"]),
                    mee_pos))
            positive_nids.extend(temp[:20])
            query_nids = MapData.query_ball_points(mee_pos, 800)  # 其余300个点作为负样本
            negetive_nids.extend(query_nids)
            positive_nids = list(set(positive_nids))
            temp2 = []
            for nid in list(set(negetive_nids)):
                if nid not in positive_nids:
                    temp2.append(nid)
            negetive_nids = temp2

            # 对该轨迹的该cellular，构造训练数据
            for i in range(5):
                neg_sample=random.sample(negetive_nids,len(positive_nids))
                train_data.append([dgl_cellular_id,positive_nids,neg_sample])
        for i in range(len(train_data)):
            temp=train_data[i]
            temp.append(dgl_cellular_id_lst)
            shortcuts_train_data[traj_id+f"~{i}"]=temp

    print("保存gcn_new_train_data对象")
    with open(opt.gcn_new_train_data_path, mode='wb') as f:
        pickle.dump(shortcuts_train_data, f)


if __name__ == "__main__":
    export_networkx_obj()
