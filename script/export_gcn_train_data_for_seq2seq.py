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
with open("../data/relationship2roadID.obj", mode='rb') as f:
    relationship2roadID = pickle.load(f)

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
        gcn_nxgraph.add_edge(
            edge_src,
            edge_dst,
            relationship=edge_relationship,
            type="node2node",
            gcn_weight=1
        )


    max_id = max(MapData.nxgraph.nodes)
    counter = max_id + 1
    # networkx?????????id??????road id???????????????
    dgl_cellular_counter = 0
    # dgl????????????id??????0?????????????????????
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

        # ?????????MEE????????????MEE????????????????????????(???????????????MEE???????????????????????????GPS????????????)
        gps_matched_routes_str = line["gpsmatch_routes"]
        gps_matched_routes_lst = gps_matched_routes_str.split(",")

        # ??????????????????edge
        for i in range(len(gps_matched_routes_lst)):
            src_road = gps_matched_routes_lst[i]
            roadID=relationship2roadID[src_road]


            if(i<=len(gps_matched_routes_lst)-2):
                edge_node2node_src_list.append(roadID)
                edge_node2node_dst_list.append(relationship2roadID[gps_matched_routes_lst[i+1]])

            for j in range(i + 1, min(i + 15, len(gps_matched_routes_lst))):  # ?????????15???????????????
                matched_road = gps_matched_routes_lst[j]
                matched_roadID = relationship2roadID[matched_road]
                edge_road2road_trans_src_list.append(roadID)
                edge_road2road_trans_dst_list.append(matched_roadID)

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

            temp = []
            for j in range(len(gps_matched_routes_lst)):
                matched_road = gps_matched_routes_lst[j]
                matched_roadID= relationship2roadID[matched_road]
                temp.append([
                    matched_roadID,
                    (MapData.nxgraph.nodes[MapData.rid2nids[matched_road][0]]["lng"],
                     MapData.nxgraph.nodes[MapData.rid2nids[matched_road][0]]["lat"])
                ])

            temp.sort(
                key=lambda x: MapData.get_distance(x[1],
                                                   mee_pos))
            temp=[item[0] for item in  temp]
            temp = temp[:20]  # ?????????????????????20???node?????????cellular?????????????????????node????????????????????????
            positive_nids.extend(temp)
            query_nids_points = MapData.query_ball_points((mee_lng, mee_lat), 1000)  # ??????2000?????????????????????
            query_nids=[]
            for item in query_nids_points:
                rid_lst=MapData.nid2rids[item]
                for rid in rid_lst:
                    if(rid in relationship2roadID):
                        query_nids.append(relationship2roadID[rid])
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

        # ?????????????????????????????????
        positive_nids = list(set(positive_nids))
        temp2 = []
        for nid in list(set(negetive_nids)):
            if nid not in positive_nids:
                temp2.append(nid)
        negetive_nids = temp2
        for i in range(15):
            neg_sample = random.sample(negetive_nids, len(positive_nids))
            gcn_train_data[traj_id + f"~{i}"] = [dgl_cellular_id_lst, positive_nids, neg_sample]

    gcn_node_index_lst = sorted(list(gcn_nxgraph.nodes()))
    node2index_dict = {}
    for idx in range(len(gcn_node_index_lst)):
        node2index_dict[gcn_node_index_lst[idx]] = idx

    # ??????dgl?????????
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

    print("??????dgl?????????")
    dgl.save_graphs(opt.dgl_graph_path, dgl_graph)

    print("??????pos2dgl_cellular_id??????")
    with open(opt.pos2dgl_cellular_id_path, mode='wb') as f:
        pickle.dump(pos2dgl_cellular_id, f)

    print("??????dgl_cellular_id2pos??????")
    with open(opt.dgl_cellular_id2pos_path, mode='wb') as f:
        pickle.dump(dgl_cellular_id2pos, f)

    print("??????gcn_nxgraph??????")
    with open(opt.gcn_nxgraph_path, mode='wb') as f:
        pickle.dump(gcn_nxgraph, f)

    print("??????pos2cellular_id??????")
    with open(opt.pos2cellular_id_path, mode='wb') as f:
        pickle.dump(pos2cellular_id, f)

    print("??????gcn_train_data??????")
    with open(opt.gcn_train_data_path, mode='wb') as f:
        pickle.dump(gcn_train_data, f)

    print("??????node2index_dict??????")
    with open(opt.node2index_dict_path, mode='wb') as f:
        pickle.dump(node2index_dict, f)


if __name__ == "__main__":
    export_networkx_obj()
