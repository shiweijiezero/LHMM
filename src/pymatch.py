import networkx
import os
import pickle
import sys
base_path = os.path.abspath("../")
sys.path.append(base_path)
import math
import numpy as np
from utils.mydb import MYDB
from utils.mymap import MapDataset
from utils.mygps import GPSDataset
from utils.mymee import MEEDataset
from utils.mybase import dis_between_pos
import config

MapData=MapDataset()
GPSData=GPSDataset()
MEEData=MEEDataset()

print(max(MapData.nxgraph.nodes.keys()))
print(len(MapData.nxgraph.nodes.keys()))

def gaussian(sigma, x, u):
    # y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
    y = 1.0 / sigma * math.exp(-(x - u) / sigma)
    return y


def emit_prob(observed_pos, hidden_state_pos, opt):
    dis = dis_between_pos(observed_pos, hidden_state_pos)
    # print(f"DIS:{dis}")
    return gaussian(opt.gps_sigmaM, dis, 0)
    # up = math.exp(-1 * (dis_between_pos(observed_pos, hidden_state_pos) ** 2) / (2 * opt.sigmaM ** 2))
    # down = (2 * math.pi) ** 0.5 * opt.sigmaM
    #
    # if (up == 0):
    #     up = 0.00001
    # return up / down


def route_dis_by_nid(prev_nid, this_nid):
    a, b = (prev_nid, this_nid) if prev_nid < this_nid else (this_nid, prev_nid)
    try:
        dis = networkx.astar_path_length(MapData.nxgraph, source=a, target=b, weight='dist')
        # dis = dis_between_pos(dbsession.get_node_pos_by_nid(a), dbsession.get_node_pos_by_nid(b))
    except:
        print(f"{prev_nid}-{this_nid}的寻路失败")
        dis = 1000000

    return dis


def trans_prob(prev_nid, this_nid, opt, cache=None):
    # global floyd_matrix
    # if(floyd_matrix is None):
    #     floyd_matrix = networkx.floyd_warshall(nxgraph)
    route_dis = route_dis_by_nid(prev_nid, this_nid)
    dis_between = dis_between_pos(MapData.get_pos_by_node_id(prev_nid), MapData.get_pos_by_node_id(this_nid))
    prob = gaussian(opt.gps_sigmaM, route_dis, dis_between)
    # print(f"routedis:{route_dis},transprob:{prob}")
    return prob


def mymatch(gps_pos_lines: list,opt=config.DefaultConfig()):
    for gps_pos_line in gps_pos_lines:
        print(f"Len of this line: {len(gps_pos_line)}")


        # candidate_layers = [MAP.query_by_pos_lines([[pos]], k=canditate_range, return_type='nids') for pos in
        #                     gps_pos_line]
        candidate_layers = [MapData.query_ball_points(pos, opt.gps_candidate_range) for pos in gps_pos_line]

        prob_matrix = []

        for layer_id in range(len(candidate_layers)):
            print(f"Constructing {layer_id}-th layer...")
            prob_matrix.append({})
            this_pos = gps_pos_line[layer_id]
            layer = candidate_layers[layer_id]
            for canditate_nid in layer:
                candidate_pos = MapData.get_pos_by_node_id(canditate_nid)

                eprob = emit_prob(this_pos, candidate_pos, opt)

                if (layer_id == 0):
                    # 第一层 HMM
                    prob_matrix[layer_id][canditate_nid] = {'prob': eprob,
                                                            'path': [canditate_nid]}
                else:
                    # prob_matrix[layer_id][canditate_nid] = {'prob': 0, 'path': []}
                    max_prob = 0
                    max_path = []
                    for prev_nid in prob_matrix[layer_id - 1].keys():
                        new_prob = trans_prob(prev_nid=prev_nid, this_nid=canditate_nid, opt=opt) * \
                                   eprob
                        new_prob = prob_matrix[layer_id - 1][prev_nid]['prob'] + new_prob
                        # prob = trans_prob(prev_nid=prev_nid, this_nid=canditate_nid, opt=opt) * \
                        #        prob_matrix[layer_id - 1][prev_nid]['prob']
                        if (new_prob > max_prob):
                            max_prob = new_prob
                            max_path = prob_matrix[layer_id - 1][prev_nid]['path'] + [canditate_nid]
                    prob_matrix[layer_id][canditate_nid] = {'prob': max_prob,
                                                            'path': max_path}
                # print(f"{layer_id},{canditate_nid},{prob_matrix[layer_id][canditate_nid]}")

        LAST_LAYER = len(candidate_layers) - 1

        # 遍历最后一层
        max_prob = 0
        max_path = []
        for canditate_nid in candidate_layers[-1]:
            if prob_matrix[LAST_LAYER][canditate_nid]['prob'] > max_prob:
                max_prob = prob_matrix[LAST_LAYER][canditate_nid]['prob']
                max_path = prob_matrix[LAST_LAYER][canditate_nid]['path']

        # rst += max_path
        rst = set()
        rst2 = set()

        for prev_nid, this_nid in zip(max_path[:-1], max_path[1:]):
            try:
                seg_nids = networkx.shortest_path(MapData.nxgraph, prev_nid, this_nid, 'dist')
                for prev_nid_in_seg, this_nid_in_seg in zip(seg_nids[:-1], seg_nids[1:]):
                    seg_pos_tuple = MapData.get_pos_by_node_id(prev_nid_in_seg), MapData.get_pos_by_node_id(
                        this_nid_in_seg)
                    rst.add(seg_pos_tuple)
                    try:
                        rst2.add(MapData.nxgraph[prev_nid_in_seg][this_nid_in_seg]["relationship"])
                    except:
                        print("no relationship~!")
            except:
                print(f"No path from {prev_nid} ot {this_nid}")
            # try:
            #     rst.append([MAP.get_node_by_id(nid) for nid in networkx.shortest_path(nxgraph, prev_nid, this_nid, 'length')])
            # except:
            #     print(f"No path from {prev_nid} ot {this_nid}")
    # return list(rst)  # [rids]
    return list(rst),list(rst2)


if __name__=="__main__":
    all_id=GPSData.get_all_id()
    GPS_lines=[]
    for id in all_id[:1]:
        print(id)
        line=GPSData.get_gps_line_without_ts(id)
        GPS_lines.append(line)

    # 由于 MapData中，一条路的中间节点没有被包含在edge中，所以会导致nodes中的节点可能不出现在edge中，
    # 使得GPS匹配最短路径或者由node1-node2寻找其对于的relationship不存在！
    res,res2=mymatch(GPS_lines)
    print(res)
    print(res2)