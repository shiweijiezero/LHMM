import os
import pickle
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
import random

opt = DefaultConfig()


def export_networkx_obj():
    MapData = mymap.MapDataset()
    gcn_nxgraph = networkx.DiGraph()

    gcn_train_data = {}
    pos2cellular_id = {}
    pos2dgl_cellular_id = {}
    dgl_cellular_id2pos = {}

    edge_node2node_src_list = []
    edge_node2node_dst_list = []
    edge_cellular2cellular_src_list = []
    edge_cellular2cellular_dst_list = []
    edge_node2cellular_src_list = []
    edge_node2cellular_dst_list = []
    edge_cellular2node_src_list = []
    edge_cellular2node_dst_list = []
    edge_road2road_trans_src_list = []
    edge_road2road_trans_dst_list = []

    mongo_db = mydb.MYDB().db
    network_e_table = mongo_db[opt.roadnetwork_e_mongo_table]
    match_result_table = mongo_db[opt.match_result_table_path]
    edge_lines = network_e_table.find()
    match_result_lines = match_result_table.find()

    for edge_line in tqdm(edge_lines):
        edge_relationship = edge_line["relationship"]
        edge_src = edge_line["src"]
        edge_dst = edge_line["dst"]
        edge_dist = edge_line["dist"]
        edge_Name = edge_line["Name"]
        edge_maxSpeed = edge_line["maxSpeed"]
        edge_way = edge_line["way"]
        edge_forward_access = edge_line["forward_access"]
        edge_backward_access = edge_line["backward_access"]

        # if(edge_maxSpeed<opt.roadSpeedFilterPara and edge_Name=="unnamed"):
        #     continue
        edge_node2node_src_list.append(edge_src)
        edge_node2node_dst_list.append(edge_dst)

        gcn_nxgraph.add_edge(
            edge_src,
            edge_dst,
            relationship=edge_relationship,
            type="node2node",
            gcn_weight=1
        )

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
            lng = round(float(lng), 5)
            lat = round(float(lat), 5)
            if ((lng, lat) not in pos2cellular_id):
                pos2cellular_id[(lng, lat)] = counter
                counter += 1
            if ((lng, lat) not in pos2dgl_cellular_id):
                pos2dgl_cellular_id[(lng, lat)] = dgl_cellular_counter
                dgl_cellular_id2pos[dgl_cellular_counter] = (lng, lat)
                dgl_cellular_counter += 1

        for prev_pos_str, this_pos_str in zip(filtered_traj_mee_lst[:-1], filtered_traj_mee_lst[1:]):
            prev_pos = prev_pos_str.split()
            prev_pos[0], prev_pos[1] = float(prev_pos[0]), float(prev_pos[1])
            prev_pos = tuple(prev_pos)
            this_pos = this_pos_str.split()
            this_pos[0], this_pos[1] = float(this_pos[0]), float(this_pos[1])
            this_pos = tuple(this_pos)
            prev_id = pos2cellular_id[prev_pos]
            this_id = pos2cellular_id[this_pos]
            prev_dgl_id = pos2dgl_cellular_id[prev_pos]
            this_dgl_id = pos2dgl_cellular_id[this_pos]

            edge_cellular2cellular_src_list.append(prev_dgl_id)
            edge_cellular2cellular_dst_list.append(this_dgl_id)

            if (gcn_nxgraph.has_edge(prev_id, this_id)):
                gcn_nxgraph[prev_id][this_id]["gcn_weight"] = gcn_nxgraph[prev_id][this_id]["gcn_weight"] + 1
            else:
                gcn_nxgraph.add_edge(
                    prev_id,
                    this_id,
                    type="cellular2cellular",
                    gcn_weight=1
                )

        # 可以让MEE点与所有MEE匹配路径点形成边(这里可以是MEE匹配结果，也可以是GPS匹配结果)
        gps_matched_routes_str = line["gpsmatch_routes"]
        gps_matched_routes_lst = gps_matched_routes_str.split(",")

        # 构建道路转移edge
        for i in range(len(gps_matched_routes_lst)):
            src_road = gps_matched_routes_lst[i]
            src_nodes_start, src_nodes_end = MapData.rid2nids[src_road]
            edge_road2road_trans_src_list.append(src_nodes_start)
            edge_road2road_trans_dst_list.append(src_nodes_end)
            for j in range(i + 1, min(i + 15, len(gps_matched_routes_lst))):  # 最多取15个道路转移
                matched_road = gps_matched_routes_lst[j]
                matched_nodes = MapData.rid2nids[matched_road]
                node_src, node_dst = matched_nodes[0], matched_nodes[1]
                edge_road2road_trans_src_list.append(src_nodes_start)
                edge_road2road_trans_dst_list.append(node_dst)

        positive_nids = []
        negetive_nids = []
        dgl_cellular_id_lst = []
        for i in range(len(filtered_traj_mee_lst)):
            traj = filtered_traj_mee_lst[i]
            mee_lng, mee_lat = traj.split()
            mee_pos = (float(mee_lng), float(mee_lat))

            cellular_id = pos2cellular_id[mee_pos]
            dgl_cellular_id = pos2dgl_cellular_id[mee_pos]
            dgl_cellular_id_lst.append(dgl_cellular_id)

            temp1 = []
            for j in range(len(gps_matched_routes_lst)):
                matched_road = gps_matched_routes_lst[j]
                matched_nodes = MapData.rid2nids[matched_road]
                node_src, node_dst = matched_nodes[0], matched_nodes[1]
                temp1.append(node_src)
                temp1.append(node_dst)
            # 保序去重复
            temp = []
            for i in temp1:
                if not i in temp:
                    temp.append(i)

            temp.sort(
                key=lambda x: MapData.get_distance((MapData.nxgraph.nodes[x]["lng"], MapData.nxgraph.nodes[x]["lat"]),
                                                   mee_pos))
            temp = temp[:20]  # 每个基站只保留20个node节点与cellular相连，并把这些node合并起来作为正例
            positive_nids.extend(temp)
            query_nids = MapData.query_ball_points((mee_lng, mee_lat), 2000)  # 其余2000个点作为负样本
            negetive_nids.extend(query_nids)

            for node_id in temp:
                edge_node2cellular_src_list.append(node_id)
                edge_node2cellular_dst_list.append(dgl_cellular_id)
                edge_cellular2node_src_list.append(dgl_cellular_id)
                edge_cellular2node_dst_list.append(node_id)
                if (gcn_nxgraph.has_edge(node_id, cellular_id)):
                    gcn_nxgraph[node_id][cellular_id]["gcn_weight"] = gcn_nxgraph[node_id][cellular_id][
                                                                          "gcn_weight"] + 1
                    gcn_nxgraph[cellular_id][node_id]["gcn_weight"] = gcn_nxgraph[cellular_id][node_id][
                                                                          "gcn_weight"] + 1
                else:
                    gcn_nxgraph.add_edge(
                        node_id,
                        cellular_id,
                        type="node2cellular",
                        gcn_weight=1
                    )
                    gcn_nxgraph.add_edge(
                        cellular_id,
                        node_id,
                        type="cellular2node",
                        gcn_weight=1
                    )

        # 对这个轨迹，构造正反例
        positive_nids = list(set(positive_nids))
        temp2 = []
        for nid in list(set(negetive_nids)):
            if nid not in positive_nids:
                temp2.append(nid)
        negetive_nids = temp2
        for i in range(25):
            neg_sample = random.sample(negetive_nids, len(positive_nids))
            gcn_train_data[traj_id + f"~{i}"] = [dgl_cellular_id_lst, positive_nids, neg_sample]

    gcn_node_index_lst = sorted(list(gcn_nxgraph.nodes()))
    node2index_dict = {}
    for idx in range(len(gcn_node_index_lst)):
        node2index_dict[gcn_node_index_lst[idx]] = idx

    # 构建dgl异构图
    graph_data = {
        ('road', 'road2road', 'road'):
            (torch.tensor(edge_node2node_src_list), torch.tensor(edge_node2node_dst_list)),
        ('cellular', 'cellular2cellular', 'cellular'):
            (torch.tensor(edge_cellular2cellular_src_list), torch.tensor(edge_cellular2cellular_dst_list)),
        ('road', 'road2cellular', 'cellular'):
            (torch.tensor(edge_node2cellular_src_list), torch.tensor(edge_node2cellular_dst_list)),
        ('cellular', 'cellular2road', 'road'):
            (torch.tensor(edge_cellular2node_src_list), torch.tensor(edge_cellular2node_dst_list)),
        ('road', 'travel', 'road'):
            (torch.tensor(edge_road2road_trans_src_list), torch.tensor(edge_road2road_trans_dst_list)),
    }
    dgl_graph = dgl.heterograph(graph_data)
    print(dgl_graph.ntypes)
    print(dgl_graph.etypes)
    print(dgl_graph.canonical_etypes)
    print(dgl_graph.num_nodes('road'))

    print("保存dgl图对象")
    dgl.save_graphs(opt.dgl_graph_path, dgl_graph)

    print("保存pos2dgl_cellular_id对象")
    with open(opt.pos2dgl_cellular_id_path, mode='wb') as f:
        pickle.dump(pos2dgl_cellular_id, f)

    print("保存dgl_cellular_id2pos对象")
    with open(opt.dgl_cellular_id2pos_path, mode='wb') as f:
        pickle.dump(dgl_cellular_id2pos, f)

    print("保存gcn_nxgraph对象")
    with open(opt.gcn_nxgraph_path, mode='wb') as f:
        pickle.dump(gcn_nxgraph, f)

    print("保存pos2cellular_id字典")
    with open(opt.pos2cellular_id_path, mode='wb') as f:
        pickle.dump(pos2cellular_id, f)

    print("保存gcn_train_data字典")
    with open(opt.gcn_train_data_path, mode='wb') as f:
        pickle.dump(gcn_train_data, f)

    print("保存node2index_dict字典")
    with open(opt.node2index_dict_path, mode='wb') as f:
        pickle.dump(node2index_dict, f)


if __name__ == "__main__":
    export_networkx_obj()
