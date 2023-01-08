import os
import pickle
import sys
base_path = os.path.abspath("../")
sys.path.append(base_path)
import networkx
import pyprind

from config import DefaultConfig
from utils import mydb


opt = DefaultConfig()


def export_networkx_obj():
    nxgraph = networkx.DiGraph()

    nid2rids = {}
    pos2nid = {}
    rid2nids = {}

    mongo_db = mydb.MYDB().db
    network_e_table = mongo_db[opt.roadnetwork_e_mongo_table]
    network_v_table = mongo_db[opt.roadnetwork_v_mongo_table]
    edge_lines = network_e_table.find()
    vertices_lines = network_v_table.find()

    num_v = network_v_table.count()
    num_e = network_e_table.count()
    print(num_v, num_e)

    # 占位符，用于保存node_id行对应的经纬坐标，进而生成KDtree
    # print(max_id)
    # poss = np.zeros((max_id + 1, 2))
    # 这里是错误的！ 因为vertices_lines迭代器已经被遍历过了，这次已经成空的了！
    # for vertices_line in vertices_lines:
    #     node_id = vertices_line["id"]
    #     node_lat = vertices_line["Latitude"]
    #     node_lng = vertices_line["Longitude"]
    #     poss[node_id, :] = node_lng, node_lat
    # print("保存KDtree对象")
    # kdtree = KDTree(data=poss)
    # with open(opt.kdtree_path, mode='wb') as f:
    #     pickle.dump(kdtree, f)

    bar = pyprind.ProgBar(iterations=num_e)
    for edge_line in edge_lines:
        bar.update()
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

        for nid in [edge_src, edge_dst]:
            if (nid not in nid2rids):
                nid2rids[nid] = set()
            nid2rids[nid].add(edge_relationship)
            if (edge_relationship not in rid2nids):
                rid2nids[edge_relationship] = []
            rid2nids[edge_relationship].append(nid)

        nxgraph.add_edge(
            edge_src,
            edge_dst,
            relationship=edge_relationship,
            distance=edge_dist,
            name=edge_Name,
            maxSpeed=edge_maxSpeed,
            way=edge_way,
            forward_access=edge_forward_access,
            backward_access=edge_backward_access
        )

    # 这里直接通过添加边来添加节点，去除孤立中间节点的添加
    bar = pyprind.ProgBar(iterations=num_v)
    for vertices_line in vertices_lines:
        bar.update()
        node_id = vertices_line["id"]
        node_lat = vertices_line["Latitude"]
        node_lng = vertices_line["Longitude"]

        if (nxgraph.has_node(node_id)):
            nxgraph.nodes[node_id].update({"lat": node_lat, "lng": node_lng})
            pos2nid[(round(node_lng,5), round(node_lat,5))] = node_id

    print("保存nxgraph对象")
    with open(opt.nxgraph_path, mode='wb') as f:
        pickle.dump(nxgraph, f)

    print("保存nid2rids字典")
    with open(opt.nid2rids_path, mode='wb') as f:
        pickle.dump(nid2rids, f)
    print("保存pos2nid字典")
    with open(opt.pos2nid_path, mode='wb') as f:
        pickle.dump(pos2nid, f)
    print("保存rid2nids字典")
    with open(opt.rid2nids_path, mode='wb') as f:
        pickle.dump(rid2nids, f)


if __name__ == "__main__":
    export_networkx_obj()
