import os
import pickle
from numpy import array
from config import DefaultConfig
from scipy.spatial.kdtree import KDTree
import networkx
import sys
import dgl


class GCN_graph():
    def __init__(self, opt=DefaultConfig()):
        self.opt = opt
        self.pos2cellular_id = {}

        print("Reading dgl_graph file...")
        self.dgl_graph, _ = dgl.load_graphs(opt.dgl_graph_path)
        self.dgl_graph = self.dgl_graph[0]

        with open(opt.pos2dgl_cellular_id_path, mode='rb') as f:
            print("Reading pos2dgl_cellular_id file...")
            self.pos2dgl_cellular_id: dict = pickle.load(f)

        with open(opt.dgl_cellular_id2pos_path, mode='rb') as f:
            print("Reading dgl_cellular_id2pos file...")
            self.dgl_cellular_id2pos: dict = pickle.load(f)

        with open(opt.pos2cellular_id_path, mode='rb') as f:
            print("Reading pos2cellular_id file...")
            self.pos2cellular_id: dict = pickle.load(f)

        with open(opt.gcn_nxgraph_path, mode='rb') as f:
            print("Reading gcn_nxgraph file...")
            self.gcn_nxgraph: networkx.DiGraph = pickle.load(f)

        with open(opt.gcn_train_data_path, mode='rb') as f:
            print("Reading gcn_train_data file...")
            self.gcn_train_data: dict = pickle.load(f)

        with open(opt.node2index_dict_path, mode='rb') as f:
            print("Reading node2index_dict file...")
            self.node2index_dict: dict = pickle.load(f)


if __name__ == "__main__":
    base_path = os.path.abspath("../")
    sys.path.append(base_path)
    gcn_graph = GCN_graph()
